# VectorNova — Explore Vector Databases and RAG in Python

A Python-based **vector database** project with an interactive web UI. VectorNova explores similarity search using **HNSW**, **KD-Tree**, and **Brute Force**, and includes a document-based **Retrieval-Augmented Generation (RAG)** pipeline powered by local models through Ollama.

> Built as a hands-on project to understand vector indexing, similarity search, document retrieval, and how local language models can answer questions using retrieved context.

---

## What This Project Does

| Feature | Description |
|---|---|
| **3 Search Approaches** | HNSW approximate search, KD-Tree search for supported distance metrics, and brute-force exact search |
| **3 Distance Metrics** | Cosine, Euclidean, and Manhattan distance |
| **Demo Vectors** | 20 preloaded 16-dimensional demo vectors for exploring the search interface |
| **Interactive Visualization** | Web interface for exploring vectors and search results |
| **Persistent Storage** | SQLite storage for vectors and uploaded document chunks |
| **Document Embeddings** | Uses Ollama's `nomic-embed-text` model to embed document chunks |
| **RAG Pipeline** | Retrieves relevant document chunks and sends them as context to `llama3.2` for answer generation |
| **Benchmarking** | Compare search approaches and inspect query latency |
| **REST API** | Endpoints for search, insertion, deletion, statistics, documents, RAG, and HNSW configuration |

---

## How It Works

```text
                         Vector Search
                         --------------
Demo vector / query  ->  VectorDB  ->  HNSW / KD-Tree / Brute Force
                                         |
                                         v
                                  Ranked search results

                       Document RAG Pipeline
                       ---------------------
Document text -> Chunking -> Ollama (nomic-embed-text) -> Vector index
                                                              |
Question -> Ollama embedding -> Similarity search -> Retrieved chunks
                                                              |
                                                              v
                                               Ollama (llama3.2)
                                                              |
                                                              v
                                                    Generated answer
```

**HNSW (Hierarchical Navigable Small World)** is an approximate nearest-neighbor search algorithm that organizes vectors in a multilayer graph. It can reduce the amount of search work needed for large collections, at the cost of potentially returning results that differ from exact search.

**KD-Tree** partitions vector space along coordinate dimensions. Its effectiveness depends on the dimensionality and data distribution; it is not always faster than a linear scan.

**Brute-force search** compares the query against every stored vector. It provides an exact baseline for evaluating approximate search results.

---

## Prerequisites

You need the following installed:

1. **Python 3.10 or compatible**
2. **Git**
3. **Ollama** for local embedding and text-generation models

The project uses Python packages including Flask, NumPy, scikit-learn, and USearch. Install the exact dependencies from `requirements.txt` if it is present in the repository.

---

## Step-by-Step Setup (Windows)

### Step 1 — Install Python

1. Download Python from **https://www.python.org/downloads/**.
2. During installation, enable **Add Python to PATH**.
3. Open PowerShell and verify:

```powershell
python --version
```

### Step 2 — Install Git

Download Git for Windows from **https://git-scm.com/download/win** and install it with the recommended settings. Verify the installation:

```powershell
git --version
```

### Step 3 — Install Ollama and pull the models

1. Download Ollama from **https://ollama.com** and install it.
2. Open PowerShell and download the embedding and generation models:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
```

3. Check that the models are available:

```powershell
ollama list
```

Ollama must be running when you use document embedding or RAG features. Depending on your installation, it may already be running in the background. If it is not, start it with:

```powershell
ollama serve
```

### Step 4 — Clone the repository

```powershell
git clone https://github.com/shaurya3000/VectorDB-Python.git
cd VectorDB-Python
```

### Step 5 — Create a virtual environment and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If `requirements.txt` is present, install the listed dependencies:

```powershell
pip install -r requirements.txt
```

If PowerShell blocks activation, you can run the environment's Python directly or follow Microsoft's guidance for PowerShell execution policies. Do not install packages globally unless you intend to.

### Step 6 — Run the application

From the project directory, run:

```powershell
python app.py
```

Open your browser at:

```text
http://127.0.0.1:8080
```

The application uses Flask for the web server. Keep the terminal open while using the application.

---

## Using the Application

### Tab 1: Search (Demo Vectors)

- Explore the preloaded demo vectors.
- Choose a search algorithm and a distance metric.
- Run a search and inspect the returned matches and latency.
- Compare search approaches using the benchmark interface.

The demo vectors are synthetic examples intended to make the search workflow easy to explore. They are not a trained semantic embedding dataset and should not be treated as evidence of real-world semantic retrieval quality.

### Tab 2: Documents

1. Enter a document title.
2. Paste or upload text using the document interface available in the app.
3. Submit the document for processing.
4. The application splits long text into overlapping chunks, creates embeddings using `nomic-embed-text`, and stores document data persistently in SQLite.
5. The document chunks are indexed for similarity retrieval.

The current chunking configuration uses approximately **250 words per chunk with a 30-word overlap**. Chunk boundaries and overlap can affect which passages are retrieved for a question.

### Tab 3: Ask AI (RAG Pipeline)

1. Insert one or more documents first.
2. Ask a question about the stored content.
3. The application embeds the question, retrieves relevant chunks, and sends the retrieved context to `llama3.2`.
4. The language model generates an answer based on the context supplied to it.

A RAG answer can still be incorrect even when relevant passages are retrieved. For evaluation, inspect the retrieved chunks as well as the generated answer, and test questions whose answers are absent from the documents.

---

## REST API Reference

The Flask server runs locally at `http://127.0.0.1:8080` by default.

### Vector Search Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search` | Search demo vectors using a query vector and selected options |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/<id>` | Delete a demo vector by ID |
| `GET` | `/items` | List demo vectors |
| `GET` | `/benchmark` | Compare available search approaches |
| `GET` | `/hnsw-info` | Return available HNSW information |
| `GET` | `/hnsw-config` | Read the HNSW search-expansion setting |
| `POST` | `/hnsw-config` | Update the HNSW search-expansion setting at runtime |
| `GET` | `/stats` | Return vector database statistics |
| `GET` | `/status` | Report Ollama/model status |

### Document and RAG Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/doc/insert` | Add a document and process its chunks and embeddings |
| `GET` | `/doc/list` | List stored documents |
| `DELETE` | `/doc/delete/<id>` | Delete a stored document using its ID |
| `POST` | `/doc/search` | Search document chunks by similarity |
| `POST` | `/doc/ask` | Retrieve relevant context and generate an answer |

The exact request fields and response formats are defined by the route handlers in `app.py`. Check those handlers before relying on an endpoint in an external client.

### Example: Check HNSW Configuration

```powershell
curl http://127.0.0.1:8080/hnsw-config
```

Example request to change the search-expansion setting:

```powershell
curl -X POST http://127.0.0.1:8080/hnsw-config `
  -H "Content-Type: application/json" `
  -d '{"expansionSearch":128}'
```

The setting applies at runtime. Unless it is saved separately by the application, a restart returns it to the default configured value.

### Example: Ask a Question

```powershell
curl -X POST http://127.0.0.1:8080/doc/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"What does the document say about its main topic?","k":3}'
```

---

## Project Structure

```text
VectorDB-Python/
├── app.py              # Flask application and REST API routes
├── vector_db.py        # Vector storage interface and similarity search
├── document_db.py      # Document chunking, embeddings, and retrieval
├── ollama_client.py    # Client for local Ollama models
├── storage.py          # SQLite-backed persistence
├── index.html          # Web interface
├── requirements.txt    # Python dependencies, if included
├── tests/              # Automated tests, if included
└── README.md           # Project documentation
```

File names and optional folders can change as the project evolves; use the current repository contents as the source of truth.

### Architecture

```text
Flask Web UI / REST API
          |
          +-----------------------+
          |                       |
          v                       v
      VectorDB                 DocumentDB
          |                       |
          v                       +--> Chunking
  HNSW / KD-Tree /               +--> Ollama embeddings
  Brute-force search             +--> Similarity retrieval
          |                       |
          v                       v
  Vector storage (SQLite)    Document storage (SQLite)
                                  |
                                  v
                           Ollama language model
```

- **VectorDB** manages demo vectors and search operations.
- **DocumentDB** manages document text, chunks, embeddings, and document retrieval.
- **OllamaClient** communicates with local embedding and generation models.
- **SQLite** provides persistence for stored vectors and documents.
- **Flask** exposes the user interface and API endpoints.

---

## Algorithm Deep Dive

### HNSW (Hierarchical Navigable Small World)

HNSW organizes vectors into a multilayer graph. Search begins in a sparse upper layer and moves toward increasingly dense layers to find candidate neighbors. This often reduces the amount of work compared with scanning every vector, but it is approximate: results can differ from exact nearest-neighbor search.

The `expansionSearch` setting controls the search effort exposed by the application's HNSW configuration endpoint. Higher values may improve recall while increasing latency. The best setting depends on the dataset and query workload.

### KD-Tree (K-Dimensional Tree)

A KD-Tree recursively partitions points along coordinate dimensions. Search can skip regions that cannot contain a closer point under the supported metric. KD-Trees can work well for lower-dimensional data, but their advantage often decreases as dimensionality grows.

### Brute-Force Search

Brute-force search computes the distance from the query to every stored vector and selects the nearest results. It is straightforward and exact for the selected distance metric, making it a useful baseline for checking approximate-search recall. Its work generally grows with the number of vectors and vector dimensions.

### Distance Metrics

- **Cosine distance/similarity:** Compares vector directions and is commonly used with text embeddings.
- **Euclidean distance:** Measures straight-line distance between vectors.
- **Manhattan distance:** Sums the absolute differences across dimensions.

The metric used for search should match the assumptions of the embedding model and the intended retrieval task. Scores from different metrics are not directly interchangeable.

---

## Benchmarking and Evaluation

VectorNova includes search benchmarking and can be evaluated against exact brute-force results. In one recorded synthetic-data experiment with 100,000 vectors of 16 dimensions, HNSW achieved **99.2% Recall@5**. In that run, mean query latency was approximately **310 microseconds for HNSW** and **839 microseconds for brute-force search**.

These figures are experiment-specific, not performance guarantees. Results vary with the data, hardware, search parameters, and query set. Synthetic vectors do not establish semantic quality for real documents.

For a meaningful evaluation:

1. Use the same queries and dataset for each search approach.
2. Compare approximate results with exact brute-force top-k results.
3. Measure recall as well as latency.
4. Run repeated trials and report median and p95 latency.
5. For RAG, separately measure retrieval quality, answer correctness, and whether claims are supported by retrieved context.

---

## Common Issues

| Problem | What to check |
|---|---|
| Ollama is unavailable | Confirm Ollama is installed and running; check `ollama list` |
| Embedding request fails | Confirm `nomic-embed-text` is installed and the Ollama service is reachable |
| Generation request fails | Confirm `llama3.2` is installed and the Ollama service is reachable |
| Python package import fails | Activate the virtual environment and install the project's dependencies |
| Port 8080 is already in use | Stop the process using that port or change the Flask port configuration |
| Search results look unrelated | Inspect the query embedding, metric, indexed data, and retrieved chunks |
| RAG answer is incorrect | Check whether the right chunks were retrieved before changing the prompt or generation model |
| HNSW recall is too low | Increase search expansion gradually and measure recall/latency trade-offs |
| Deleted content still appears | Check document deletion, vector-index cleanup, and persistent storage state |

### Use a Different Ollama Model

The embedding model and generation model serve different purposes. If you change the generation model, ensure the selected model is available in Ollama and update the model setting used by `ollama_client.py`. If you change the embedding model, re-embed existing documents consistently; vectors generated by different embedding models should not be mixed blindly in the same index.

---

## Limitations

- The preloaded demo vectors are synthetic and are intended for exploring search behaviour, not proving semantic understanding.
- HNSW is approximate; results can differ from exact search.
- KD-Tree performance depends on dimensionality and data distribution.
- RAG answers can be incorrect or unsupported even when retrieval returns relevant passages.
- Benchmark results depend on hardware, parameters, and dataset characteristics.
- The application is designed as an educational and portfolio project, not as a production-hardened multi-user vector database service.

---

## License

Add or update this section to match the license file included in the repository. Do not assume a license unless one is actually provided.
