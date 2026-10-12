# VectorNova — Vector Database & RAG

VectorNova is a Python-based learning and portfolio project exploring nearest-neighbor search, vector similarity, interactive visualization, and Retrieval-Augmented Generation (RAG).

## Features

- Vector insertion, retrieval, and deletion
- Brute-force nearest-neighbor search
- KD-Tree search for supported metrics
- HNSW approximate nearest-neighbor search using USearch
- Cosine, Euclidean, and Manhattan distance metrics
- Search latency measurements and algorithm comparison
- Flask web interface with vector visualization
- Document chunking, embedding, retrieval, and Ollama-powered answers
- SQLite persistence for vector records and document chunks

## Stack

Python, Flask, NumPy, scikit-learn, USearch, SQLite, Ollama, HTML, CSS, and JavaScript.

## Prerequisites

- Python 3.10 or compatible
- Ollama for document embedding and RAG

Install the models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

## Installation

```powershell
git clone https://github.com/shaurya3000/VectorDB-Python.git
cd VectorDB-Python
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run:

```powershell
python app.py
```

Open http://127.0.0.1:8080.

## Search behavior

- Brute force ranks all stored vectors exactly for the selected metric.
- KD-Tree acceleration is used for Euclidean distance; unsupported combinations fall back to exact ranking.
- HNSW uses USearch for cosine nearest-neighbor search.
- The current `/hnsw-info` endpoint does not expose actual HNSW adjacency edges; it must not be interpreted as a graph-edge visualization.

## RAG

Document text is split into overlapping chunks, embedded with `nomic-embed-text`, retrieved by vector similarity, and supplied as context to `llama3.2`.

The demo vectors use synthetic category-oriented embeddings and are not semantic embeddings of the text. General semantic retrieval is provided by the separate document/RAG path when Ollama is available.

## Persistence

SQLite stores vector records and document chunks in `vector_nova.db`. In-memory HNSW and KD-Tree indexes are rebuilt from saved embeddings when the app starts.

## Tests

Run the automated tests from the project root:

```powershell
python -m pytest -v
```

Tests are provided but should be run in the target environment before claiming that the application is fully verified.

## Attribution and licensing

This project was developed as a Python adaptation/learning project based on related vector-database and RAG concepts. Review and comply with any applicable upstream licenses and attribution requirements. Add a license only after confirming you have the right to license all included code and assets.

## Known limitations

- Demo vectors are synthetic.
- HNSW is approximate; evaluate recall against exact search on representative data.
- The current UI endpoint does not expose the actual HNSW adjacency graph.
- Performance depends on dataset size, hardware, and benchmark methodology.
