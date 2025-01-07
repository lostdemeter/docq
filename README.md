# Document Query CLI Tool

## Overview
A powerful CLI tool for querying and interacting with various document types using Ollama and LlamaIndex.

## Features
- Support for multiple document formats (PDF, DOCX, PPTX, HTML, Markdown, etc.)
- Interactive query mode
- Powered by Ollama LLM and HuggingFace embeddings

## Prerequisites
- Python 3.8+
- Ollama installed and running
- Hugging Face Transformers

## Installation
1. Clone the repository
```bash
git clone https://github.com/yourusername/docling-ollama-terminal.git
cd docling-ollama-terminal
```

2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
Basic usage:
```bash
python docq.py [document_path]
```

Query a specific document:
```bash
python docq.py document.md -q "What is the main topic?"
```

List supported file formats:
```bash
python docq.py -l
```

Use custom models:
```bash
python docq.py document.md -m mistral -e custom-embedding -r custom-rerank
```

### Interactive Mode
Once the document is loaded, you can interactively ask questions about its contents.

## Acknowledgments

This project was inspired by Fahd Mirza's [Docling with Ollama](https://github.com/fahdmirza/doclingwithollama) project. The primary goals are to:
- Convert the original work into a command-line tool
- Add reranking functionality

Special thanks to Fahd Mirza for the original implementation.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[Choose an appropriate license]
