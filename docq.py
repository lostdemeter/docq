import os
import argparse
from typing import Optional, Dict
from pathlib import Path
import sys

from llama_index.core import Settings, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.docling import DoclingReader
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
    ocr_reader = DoclingReader()
    
    # For text files, return None to use default reader
    return {
        ".pdf": ocr_reader,
        ".docx": ocr_reader,
        ".pptx": ocr_reader,
        ".xlsx": ocr_reader,
        ".png": ocr_reader,
        ".jpg": ocr_reader,
        ".jpeg": ocr_reader,
        ".html": ocr_reader,
        ".md": None,  # Use default reader for markdown
        ".adoc": None,  # Use default reader for AsciiDoc
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
    use_cache: bool = True
) -> VectorStoreIndex:
    """Set up the query engine with the given document."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_ext = Path(file_path).suffix.lower()
    extractors = get_file_extractor()
    
    if file_ext not in extractors:
        supported_formats = ", ".join(extractors.keys())
        raise ValueError(f"Unsupported file format: {file_ext}. Supported formats are: {supported_formats}")

    # Configure embedding model - default to local model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Set up cache directory
    persist_dir = Path(file_path).parent / f"{Path(file_path).stem}.index"

    # Check if we can use cached index
    if use_cache and persist_dir.exists():
        storage_context = StorageContext.from_defaults(
            persist_dir=str(persist_dir)
        )
        index = load_index_from_storage(storage_context=storage_context)
    else:
        # Always create a new index
        file_path = os.path.abspath(file_path)  # Convert to absolute path
        directory = os.path.dirname(file_path)
        file_ext = Path(file_path).suffix.lower()
        
        # Use different loading strategy for text files vs OCR files
        if file_ext in ['.md', '.adoc']:
            directory_loader = SimpleDirectoryReader(
                input_dir=directory,
                input_files=[file_path]
            )
        else:
            directory_loader = SimpleDirectoryReader(
                input_dir=directory,
                input_files=[file_path],
                file_extractor=extractors
            )
            
        documents = directory_loader.load_data()
        
        # Initialize storage context
        storage_context = StorageContext.from_defaults()

        # Create index
        transformations = []
        if Path(file_path).suffix.lower() == '.md':
            markdown_parser = MarkdownNodeParser()
            transformations.append(markdown_parser)
            
        index = VectorStoreIndex.from_documents(
            documents=documents,
            transformations=transformations,
            storage_context=storage_context,
            show_progress=False  # Disable progress bar
        )
        
        # Persist the index to disk only if caching is enabled
        if use_cache:
            persist_dir.mkdir(exist_ok=True)
            index.storage_context.persist(persist_dir=str(persist_dir))

    # Initialize reranker
    rerank_model_name = rerank_model or "BAAI/bge-reranker-v2-m3"
    reranker = SentenceTransformerRerank(
        model=rerank_model_name,
        top_n=3
    )

    # Set up LLM and custom prompt
    llm_model_name = llm_model or "llama3.2"
    Settings.llm = initialize_llm(model_name=llm_model_name)
    query_engine = index.as_query_engine(
        streaming=True,
        node_postprocessors=[reranker]
    )

    custom_qa_prompt = (
        "Context information is below. This includes OCR text extracted from images.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and OCR text above, answer the query in a direct and precise manner. "
        "If the query is about reading text from an image, focus on the OCR text that was detected. "
        "If you don't know the answer, say 'I don't know!'.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    prompt_template = PromptTemplate(custom_qa_prompt)
    query_engine.update_prompts({"response_synthesizer:text_qa_template": prompt_template})

    return query_engine

def print_help():
    """Print detailed help information about the CLI tool."""
    help_text = """
Document Q&A CLI Tool Help

Usage: python docq.py [FILE_PATH] [OPTIONS]

Positional Arguments:
  FILE_PATH            Path to the document file to query

Options:
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
            question = input("Question: ").strip()
            if not question:
                continue
                
            if question.lower() == 'quit':
                break

            print("Answer: ", end="", flush=True)
            response = query_engine.query(question)
            response_text = ""
            for text_chunk in response.response_gen:
                response_text += text_chunk
                print(text_chunk, end="", flush=True)
            print("\n")  # Two new lines after response
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")
            continue

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Document Q&A CLI Tool", add_help=False)
    parser.add_argument("file_path", nargs="?", help="Path to the document file to query")
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
            use_cache=not args.no_cache  # Invert the no-cache flag
        )

        # Handle single question or interactive mode
        if args.question:
            print("\nThinking...", end="", flush=True)
            response = query_engine.query(args.question)
            print("\r" + " " * 10 + "\r", end="", flush=True)  # Clear "Thinking..." text
            
            # Handle streaming response
            response_text = ""
            for text_chunk in response.response_gen:
                response_text += text_chunk
                print(text_chunk, end="", flush=True)
            print()  # New line after response
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