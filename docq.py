import os
import argparse
from typing import Optional, Dict
from pathlib import Path
import sys
from typing import Optional

from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
    load_index_from_storage,
    ServiceContext
)
from llama_index.core.node_parser import CodeSplitter
from llama_index.core.schema import TextNode
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.storage_context import StorageContext

def initialize_llm(model_name: str = "llama3.2") -> Ollama:
    """Initialize the Ollama LLM client."""
    return Ollama(model=model_name, request_timeout=120.0)

def get_file_extractor() -> Dict:
    """Get the file extractors for different document types."""
    # OCR-enabled reader for images and rich documents
    ocr_reader = SimpleDirectoryReader()
    
    # For text files, return None to use default reader
    return {
        ".pdf": ocr_reader,
        ".docx": ocr_reader,
        ".pptx": ocr_reader,
        ".xlsx": ocr_reader,
        ".xls": ocr_reader,
        ".png": ocr_reader,
        ".jpg": ocr_reader,
        ".jpeg": ocr_reader,
        ".html": ocr_reader,
        ".md": None,  # Use default reader for markdown
        ".adoc": None,  # Use default reader for AsciiDoc
        ".txt": None,  # Use default reader for text files
        ".py": None,  # Use default reader for Python files
    }

def list_supported_formats():
    """List all supported file formats."""
    extractors = get_file_extractor()
    print("Supported File Formats:")
    for ext in sorted(extractors.keys()):
        print(f"  {ext}")

def setup_query_engine(
    file_path: str, 
    embedding_model: Optional[str] = None, 
    llm_model: Optional[str] = None,
    rerank_model: Optional[str] = None,
    use_cache: bool = True,
    directory_mode: bool = False
) -> VectorStoreIndex:
    """Set up the query engine with the given document or directory."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Path not found: {file_path}")

    # Configure cache directory - use absolute path
    cache_dir = Path(os.getcwd()) / ".docq_cache"
    if directory_mode:
        persist_dir = cache_dir / Path(file_path).name
    else:
        persist_dir = cache_dir / Path(file_path).stem

    def cleanup_incomplete_cache():
        """Clean up incomplete cache files"""
        if persist_dir.exists():
            try:
                import shutil
                shutil.rmtree(persist_dir)
                print("Cleaned up incomplete cache.")
            except Exception as e:
                print(f"Warning: Failed to clean up cache: {e}")

    # Create cache directory if it doesn't exist
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Configure embedding model and settings
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
    )
    Settings.chunk_size = 512  # Reasonable chunk size for code
    Settings.chunk_overlap = 64   # Reasonable overlap
    Settings.num_output = 2048  # Reasonable output size

    try:
        # Check if we can use cached index
        required_files = ["docstore.json", "index_store.json", "vector_store/index.faiss"]
        cache_complete = all((persist_dir / f).exists() for f in required_files)
        
        if use_cache and persist_dir.exists() and cache_complete:
            print("Loading cached index...")
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(persist_dir)
                )
                index = VectorStoreIndex(
                    [],  # Empty documents list since we're loading from storage
                    storage_context=storage_context
                )
            except Exception as e:
                print(f"Warning: Failed to load cache: {e}")
                cleanup_incomplete_cache()
                raise
        else:
            if persist_dir.exists() and not cache_complete:
                print("Found incomplete cache, cleaning up...")
                cleanup_incomplete_cache()
                
            print("Building new index...")
            # Load documents
            if directory_mode:
                documents = SimpleDirectoryReader(
                    input_dir=file_path,
                    recursive=True,
                    exclude_hidden=True,
                    filename_as_id=True,
                    required_exts=[".py", ".md", ".txt", ".html", ".ts", ".pdf", ".xlsx", ".xls"],  
                    exclude=["**/.*", "**/.git/**", "**/.docq_cache/**", "**/node_modules/**"]  
                ).load_data()
            else:
                documents = SimpleDirectoryReader(
                    input_files=[file_path],
                    filename_as_id=True,
                ).load_data()

            if not documents:
                raise ValueError(f"No supported documents found in {file_path}")

            print(f"Loaded {len(documents)} documents")
            for doc in documents:
                print(f"- {doc.metadata['file_name']}")
            print()

            # Create storage context first
            if use_cache:
                try:
                    # Ensure all necessary cache directories exist
                    (persist_dir / "docstore").mkdir(parents=True, exist_ok=True)
                    (persist_dir / "index_store").mkdir(parents=True, exist_ok=True)
                    (persist_dir / "vector_store").mkdir(parents=True, exist_ok=True)
                    
                    storage_context = StorageContext.from_defaults(
                        persist_dir=str(persist_dir)
                    )
                except Exception as e:
                    print(f"Warning: Failed to create cache directories: {e}")
                    storage_context = StorageContext.from_defaults()
            else:
                storage_context = StorageContext.from_defaults()

            try:
                # Create index with storage context
                index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    show_progress=True
                )

                # Try to persist if caching is enabled
                if use_cache and storage_context:
                    index.storage_context.persist()
            except KeyboardInterrupt:
                print("\nInterrupted, cleaning up...")
                cleanup_incomplete_cache()
                raise

    except Exception as e:
        print(f"Error setting up index: {e}")
        raise

    # Initialize reranker
    rerank_model_name = rerank_model or "BAAI/bge-reranker-v2-m3"
    reranker = SentenceTransformerRerank(
        model=rerank_model_name,
        top_n=3
    )

    # Set up LLM and custom prompt
    llm_model_name = llm_model or "llama3.2"
    Settings.llm = initialize_llm(model_name=llm_model_name)

    # Create a custom prompt that emphasizes file information
    custom_qa_prompt = PromptTemplate(
        """Context information from various Python and text files is below. Each file's content is prefixed with its path.
---------------------
{context_str}
---------------------

Given the context information above, answer the following question. Pay special attention to:
1. The file paths and their contents
2. Any docstrings or comments that describe the purpose of the files
3. The actual code implementation if relevant to the question

If the question asks about specific files, be sure to mention which files you found the information in. If you don't find relevant information in the context, simply say you don't know.

Question: {query_str}

Answer: Let me help you with that."""
    )

    # Configure query engine with the custom prompt
    query_engine = index.as_query_engine(
        streaming=True,
        node_postprocessors=[reranker],
        text_qa_template=custom_qa_prompt,
        similarity_top_k=5  # Increase number of relevant chunks
    )

    return query_engine

def print_help():
    """Print detailed help information about the CLI tool."""
    help_text = """
Document Q&A CLI Tool Help

Usage: python docq.py [FILE_PATH] [OPTIONS]

Positional Arguments:
  FILE_PATH            Path to the document file or directory to query

Options:
  -d, --directory      Process entire directory of documents
  -n, --no-cache       Disable index caching (regenerate index each time)
  -h, --help           Show this help message and exit
  -l, --list-formats   List supported file formats
  -q, --question Q     Ask a single question about the document
  -e, --embedding-model MODEL 
                       Custom embedding model 
                       (default: sentence-transformers/all-MiniLM-L6-v2)
  -m, --llm-model MODEL 
                       Custom Ollama LLM model 
                       (default: llama3.2)
  -r, --rerank-model MODEL 
                       Custom reranking model 
                       (default: BAAI/bge-reranker-v2-m3)

Examples:
  # List supported file formats
  python docq.py -l

  # Query a document with a specific question
  python docq.py document.md -q "What is the main topic?"

  # Use custom models
  # Disable caching
  python docq.py document.md --no-cache
  python docq.py document.md -m mistral -e custom-embedding -r custom-rerank

Interactive Mode:
  When no question is provided, the tool enters an interactive mode 
  where you can ask multiple questions about the document.
"""
    print(help_text)

def interactive_mode(query_engine: VectorStoreIndex):
    """Run an interactive Q&A session."""
    print("\nEntering interactive mode. Type 'quit' to exit.\n")
    
    while True:
        try:
            question = input("Question: ")
            if question.lower() == 'quit':
                break
            if not question.strip():
                continue
                
            print("Answer:", end=" ")
            response = query_engine.query(question)
            print(response)
            print()  # Add blank line for readability
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            continue

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Document Q&A CLI Tool", add_help=False)
    parser.add_argument("file_path", nargs="?", help="Path to the document file or directory to query")
    parser.add_argument("-d", "--directory", action="store_true", 
                        help="Process entire directory of documents")
    parser.add_argument("-n", "--no-cache", action="store_true", 
                        help="Disable index caching (regenerate index each time)")
    parser.add_argument("-h", "--help", action="store_true", help="Show help message")
    parser.add_argument("-l", "--list-formats", action="store_true", 
                        help="List supported file formats")
    parser.add_argument("-q", "--question", type=str, 
                        help="Ask a single question about the document")
    parser.add_argument("-e", "--embedding-model", type=str, 
                        help="Custom embedding model")
    parser.add_argument("-m", "--llm-model", type=str, 
                        help="Custom Ollama LLM model")
    parser.add_argument("-r", "--rerank-model", type=str, 
                        help="Custom reranking model")

    # Parse known args to handle help and list formats first
    args, unknown = parser.parse_known_args()

    # Show help if requested
    if args.help:
        print_help()
        return 0

    # List supported formats if requested
    if args.list_formats:
        list_supported_formats()
        return 0

    # Validate file path
    if not args.file_path:
        print("Error: Please provide a file path.")
        print_help()
        return 1

    try:
        # Set up query engine with caching disabled if -n flag is used
        query_engine = setup_query_engine(
            file_path=args.file_path,
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
            rerank_model=args.rerank_model,
            use_cache=not args.no_cache,  # Invert the no-cache flag
            directory_mode=args.directory
        )

        # Handle single question or interactive mode
        if args.question:
            print("\nThinking..." + " " * 10)
            response = query_engine.query(args.question)
            print(response)
        else:
            # Enter interactive mode
            interactive_mode(query_engine)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())