# RAG Learning Assistant

An AI-powered assistant designed to retrieve and answer questions based on Wikipedia content using Haystack, FAISS, and Transformers. This project serves as a foundation for building a learning assistant with RAG (Retrieval-Augmented Generation) capabilities.

## Features
- Retrieves information from Wikipedia on specified topics.
- Embedding-based retrieval of relevant documents using FAISS.
- QA pipeline with T5 model for generating responses.

## Requirements

To run this project, ensure you have:
- Python 3.7+
- A virtual environment (recommended)

### Libraries

The following libraries are required:
- `haystack`
- `wikipedia-api`
- `transformers`
- `faiss-cpu` or `faiss-gpu` (if available)

You can install these by running:
```bash
pip install -r requirements.txt
