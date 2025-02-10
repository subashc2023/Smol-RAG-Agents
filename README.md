# RAG Documentation System

A Retrieval-Augmented Generation (RAG) system that enhances LLM responses with relevant document context.

## Features

- Efficient document chunking with context preservation
- Markdown-aware processing that maintains document structure
- Semantic search using multilingual embeddings
- Incremental document updates using content hashing
- Integration with Google's Gemini LLM

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
GEMINI_API_KEY=your_api_key_here
```

3. Place your documentation in the `docs/` directory as markdown files.

## Usage

### Basic Usage

```python
from agent import agent

# Ask questions about your documentation
response = agent.run("What is the purpose of this project?")
```

### Command Line Interface

The system provides several CLI commands for database management:

```bash
# Add/update documents
python src/embed.py

# List all stored passages
python src/embed.py list

# Clear the database
python src/embed.py clear

# Perform a search query
python src/embed.py "your search query"
```

## Architecture

- `text_chunker.py`: Handles intelligent document splitting
- `embed.py`: Manages document processing and vector database
- `retriever.py`: Implements semantic search functionality
- `agent.py`: Integrates components with LLM using smolagents

## Technical Details

- Uses Alibaba's multilingual embedding model for semantic search
- ChromaDB for vector storage
- Google Gemini for LLM responses
- Smart chunking preserves document context and header hierarchy
- Incremental updates avoid re-embedding unchanged content

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request