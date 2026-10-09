# VectorNova — Vector Database & RAG

VectorNova is a Python-based vector database project that explores nearest-neighbor search, similarity metrics, interactive visualization, and Retrieval-Augmented Generation (RAG).

## Features

- Vector insertion, retrieval, and deletion
- Brute-force nearest-neighbor search
- KD-Tree search for supported distance metrics
- HNSW approximate nearest-neighbor search using USearch
- Cosine, Euclidean, and Manhattan distance metrics
- Search latency measurement and algorithm comparison
- Interactive web interface built with Flask and JavaScript
- Vector embedding visualization
- Document chunking and vector retrieval
- Ollama integration for text embeddings and answer generation
- SQLite-based persistence for vector and document data

## Technology Stack

- **Language:** Python
- **Backend:** Flask
- **Numerical computing:** NumPy
- **Spatial indexing:** scikit-learn KD-Tree
- **Approximate nearest-neighbor search:** USearch / HNSW
- **Language models and embeddings:** Ollama
- **Frontend:** HTML, CSS, JavaScript
- **Storage:** SQLite

## Prerequisites

- Python 3.10 or a compatible Python version
- pip
- Ollama for document embeddings and AI-generated answers

Install the required Ollama models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/VectorDB-Python.git
cd VectorDB-Python
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start Ollama if you intend to use the document and RAG features.

Run the Flask application:

```bash
python app.py
```

Open the application in your browser:

http://127.0.0.1:8080

## Search Algorithms

### Brute Force

Calculates distances against stored vectors and ranks the results. This provides exact nearest-neighbor results for the selected distance metric.

### KD-Tree

Uses a spatial search structure for supported distance metrics. Unsupported algorithm and metric combinations use the exact-search fallback implemented in the project.

### HNSW

Uses a graph-based approximate nearest-neighbor index through USearch. Search performance and recall depend on the dataset and index settings.

## Distance Metrics

- **Cosine distance:** Measures the difference in vector orientation.
- **Euclidean distance:** Measures straight-line distance between vectors.
- **Manhattan distance:** Measures the sum of absolute coordinate differences.

## Retrieval-Augmented Generation (RAG)

The document workflow splits text into overlapping chunks, generates embeddings through Ollama, retrieves relevant chunks, and supplies the retrieved context to a language model to generate an answer.

The embedding model is `nomic-embed-text`, and the generation model is `llama3.2`.

## Persistence

SQLite can be used to retain vector records and document chunks between application sessions. The in-memory search indexes must be reconstructed from the stored embeddings when the application starts.

Persistence should be considered complete only when both vector records and document chunks are saved and restored correctly.

## Automated Tests

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Run the test suite:

```bash
python -m pytest -v
```

## Current Limitations

- Demo vectors use synthetic embeddings and do not provide genuine semantic search over arbitrary text.
- HNSW is an approximate search algorithm; recall should be evaluated against exact search.
- KD-Tree acceleration is used only for supported metric and algorithm combinations.
- Search latency depends on dataset size, hardware, and benchmark methodology.
- Ollama models must be installed locally to use the RAG workflow.
- Persistence and automated tests require successful integration and verification before being considered complete.

## Future Improvements

- Real semantic embeddings for general-purpose text search
- Recall@k and latency benchmarking on larger datasets
- Expanded unit and integration test coverage
- Improved error handling and observability
- Production deployment configuration
- More detailed HNSW graph visualization

## License

This project is intended for educational and portfolio purposes. Add an appropriate license after confirming that you have the right to license all included code and assets.
