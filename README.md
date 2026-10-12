# VectorNova

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
- Runtime configuration of HNSW search expansion through a Flask API
- Standalone benchmark scripts for 10K- and 100K-vector experiments

## Tech Stack

Python, Flask, NumPy, scikit-learn, USearch, SQLite, Ollama, HTML, CSS, and JavaScript.

## Prerequisites

- Python 3.10 or compatible
- Ollama for document embedding and RAG

Install the required Ollama models:

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

Run the application:

```powershell
python app.py
```

Open <http://127.0.0.1:8080> in your browser.

## Search Behavior

- **Brute force** ranks all stored vectors exactly for the selected metric.
- **KD-Tree** acceleration is used for Euclidean distance; unsupported combinations fall back to exact ranking.
- **HNSW** uses USearch for approximate cosine nearest-neighbor search.
- `/hnsw-info` reports available node metadata, but does not expose actual HNSW adjacency edges. Its response should not be interpreted as a graph-edge visualization.

## HNSW Configuration

The search-expansion parameter can be changed at runtime through the Flask API.

- `GET /hnsw-config` returns the current HNSW search-expansion value.
- `POST /hnsw-config` updates the value using a JSON request body.

Example request body:

```json
{
  "expansionSearch": 128
}
```

The default value is `64`. Higher values may improve approximate-search recall at the cost of additional search latency. The setting is held in memory and resets to the default when the application restarts.

Example PowerShell request:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8080/hnsw-config `
  -ContentType "application/json" `
  -Body '{"expansionSearch":128}'
```

## RAG

Document text is split into overlapping chunks, embedded with `nomic-embed-text`, retrieved by vector similarity, and supplied as context to `llama3.2` for answer generation.

The demo vectors use synthetic, category-oriented embeddings; they are not semantic embeddings of their associated text. Semantic retrieval is provided by the separate document/RAG path when Ollama is available.

## Persistence

SQLite stores vector records and document chunks in `vector_nova.db`. In-memory HNSW and KD-Tree indexes are rebuilt from saved embeddings when the application starts.

## Benchmark Experiments

The repository includes standalone scripts for evaluating HNSW search:

- `benchmark_hnsw.py` — evaluates search on 10,000 synthetic vectors.
- `benchmark_hnsw_100k.py` — evaluates search on 100,000 synthetic vectors.
- `benchmark_hnsw_tuning.py` — compares search-expansion settings of 64 and 128.

These experiments compare approximate HNSW results with exact cosine nearest-neighbor search. Previously observed results in the development environment included:

| Experiment | HNSW search expansion | Recall@5 | Recall@10 |
|---|---:|---:|---:|
| 100K synthetic vectors | 64 | 99.20% | 98.90% |
| 100K synthetic vectors | 128 | 99.80% | 99.90% |

In the recorded tuning run, mean HNSW query latency was approximately 161 microseconds at expansion 64 and 261 microseconds at expansion 128. These are measurements from a particular synthetic-data experiment, not production performance guarantees. Results depend on hardware, dataset, library versions, and benchmark methodology. Run the scripts in your own environment to reproduce and compare results.

## Tests

Run the automated tests from the project root:

```powershell
python -m pytest -v
```

Tests should be run in the target environment before claiming the application is fully verified.

## Attribution and Licensing

This project was developed as a learning and portfolio project. Add a license to this repository only after confirming you have the right to license all included code and assets.

## Known Limitations

- Demo vectors are synthetic and category-oriented.
- HNSW is approximate; evaluate recall against exact search on representative data.
- The `/hnsw-info` endpoint does not expose actual HNSW adjacency edges.
- Performance depends on dataset size, hardware, and benchmark methodology.
- RAG answers depend on retrieved chunks and the generation model; verify important answers against source documents.
