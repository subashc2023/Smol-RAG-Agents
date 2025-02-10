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
uv sync
```

2. Set up environment variables: You can use any model, along with the correct key. For a list of all models and providers, refer to litellm docs.
```bash
GEMINI_API_KEY=your_api_key_here
```

3. Place your documentation in the `docs/` directory as markdown files.

## Usage

### Basic Usage

```py
uv run agent.py "Describe the caves of Xylos." # optional -web to add websearch tool to agent.
```

### Command Line Interface

The embedding creation and database management is done through the CLI:

```bash
# Add/update documents and run the test query
uv run src/embed.py

# List all stored passages
uv run src/embed.py list

# Clear the database
uv run src/embed.py clear

# Perform a custom search query for testing
uv run src/embed.py "your search query"
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
