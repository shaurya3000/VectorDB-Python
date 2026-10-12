from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from vector_db import VectorDB
from document_db import DocumentDB
from ollama_client import OllamaClient
from storage import VectorStorage

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent


def make_embedding(category, seed=0):
    """Create a deterministic 16D demo vector; not a semantic text embedding."""
    rng = np.random.default_rng(seed)
    embedding = np.full(16, 0.08, dtype=np.float32)
    starts = {"cs": 0, "math": 4, "food": 8, "sports": 12}
    start = starts.get(category, 0)
    embedding[start:start + 4] = (
        np.array([0.90, 0.85, 0.75, 0.70])
        + rng.normal(0, 0.025, 4)
    )
    return np.clip(embedding, 0.01, 0.99).tolist()


demo_items = [
    ("Linked List: nodes and pointers", "cs"),
    ("Binary Search Tree: search and insert", "cs"),
    ("Dynamic Programming: overlapping subproblems", "cs"),
    ("Graph BFS and DFS traversal", "cs"),
    ("Hash Table: constant-time lookup", "cs"),
    ("Calculus: derivatives and integrals", "math"),
    ("Linear Algebra: matrices and eigenvalues", "math"),
    ("Probability and Bayes theorem", "math"),
    ("Number Theory and prime numbers", "math"),
    ("Combinatorics and permutations", "math"),
    ("Neapolitan Pizza and tomatoes", "food"),
    ("Sushi: rice and fish", "food"),
    ("Ramen noodle soup", "food"),
    ("Tacos and salsa", "food"),
    ("Croissant and pastry", "food"),
    ("Basketball: shooting and dribbling", "sports"),
    ("Football: tackles and strategy", "sports"),
    ("Tennis: serves and volleys", "sports"),
    ("Chess: tactics and strategy", "sports"),
    ("Swimming and Olympic competition", "sports"),
]

db = VectorDB(dims=16)
doc_db = DocumentDB()
ollama = OllamaClient()
storage = VectorStorage()


def restore_vectors():
    """Restore saved IDs and rebuild the in-memory indexes."""
    records = storage.load()
    for record in records:
        vector = np.asarray(record["embedding"], dtype=np.float32)
        if vector.shape != (db.dims,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid stored vector with ID {record['id']}.")
        item_id = int(record["id"])
        db.items[item_id] = {
            "id": item_id,
            "metadata": record["metadata"],
            "category": record["category"],
            "embedding": vector.tolist(),
        }
        db.hnsw.add(item_id, vector)
    if db.items:
        db.next_id = max(db.items) + 1
        db._rebuild_kdtree()


def seed_demo_vectors():
    """Create demo records only for a brand-new vector store."""
    if db.items:
        return
    for seed, (metadata, category) in enumerate(demo_items, start=1):
        vector = make_embedding(category, seed)
        item_id = db.insert(metadata, category, vector)
        storage.save(item_id, metadata, category, vector)


restore_vectors()
seed_demo_vectors()


def error_response(message, status_code=400):
    return jsonify({"error": str(message)}), status_code


def parse_query_vector():
    raw = request.args.get("v", "")
    if not raw:
        raise ValueError("Missing query vector parameter 'v'")
    try:
        vector = [float(value) for value in raw.split(",")]
    except ValueError as exc:
        raise ValueError("Query vector must contain numbers") from exc
    if not np.isfinite(np.asarray(vector, dtype=np.float32)).all():
        raise ValueError("Query vector must contain finite numbers")
    return vector


def parse_json_body():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object")
    return data


def parse_k(value, default=3):
    try:
        k = int(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("k must be an integer") from exc
    if not 1 <= k <= 100:
        raise ValueError("k must be between 1 and 100")
    return k


def embed_or_raise(text):
    embedding = ollama.embed(text)
    if not embedding:
        raise RuntimeError(
            "Could not create an embedding. Check Ollama and run: "
            "ollama pull nomic-embed-text"
        )
    vector = np.asarray(embedding, dtype=np.float32)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise RuntimeError("Ollama returned an invalid embedding.")
    return vector.tolist()


@app.get("/")
def home():
    return send_from_directory(str(BASE_DIR), "index.html")


@app.get("/items")
def items():
    return jsonify(list(db.items.values()))


@app.get("/search")
def search():
    try:
        query = parse_query_vector()
        k = parse_k(request.args.get("k", 5), 5)
        metric = request.args.get("metric", "cosine")
        algorithm = request.args.get("algo", "hnsw")
        result = db.search(query, k=k, metric=metric, algorithm=algorithm)
        return jsonify({
            "results": result["hits"],
            "latencyUs": result["latency_us"],
            "algo": algorithm,
            "metric": metric,
        })
    except (ValueError, TypeError) as exc:
        return error_response(exc)


@app.post("/insert")
def insert():
    try:
        data = parse_json_body()
        metadata = str(data.get("metadata", "")).strip()
        category = str(data.get("category", "cs"))
        embedding = data.get("embedding")
        if not metadata:
            raise ValueError("metadata is required")
        if embedding is None:
            raise ValueError("embedding is required")
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.shape != (db.dims,):
            raise ValueError(f"Embedding must contain exactly {db.dims} values")
        if not np.isfinite(vector).all():
            raise ValueError("Embedding contains invalid values")

        item_id = db.insert(metadata, category, vector)
        try:
            storage.save(item_id, metadata, category, vector.tolist())
        except Exception:
            db.delete(item_id)
            raise
        return jsonify({"id": item_id}), 201
    except (ValueError, TypeError, KeyError) as exc:
        return error_response(exc)
    except Exception:
        app.logger.exception("Vector insertion failed")
        return error_response("Could not save vector.", 500)


@app.delete("/delete/<int:item_id>")
def delete(item_id):
    if item_id not in db.items:
        return jsonify({"ok": False})
    try:
        storage.delete(item_id)
        return jsonify({"ok": db.delete(item_id)})
    except Exception:
        app.logger.exception("Vector deletion failed")
        return error_response("Could not delete vector.", 500)


@app.get("/benchmark")
def benchmark():
    try:
        query = parse_query_vector()
        k = parse_k(request.args.get("k", 5), 5)
        metric = request.args.get("metric", "cosine")
        result = db.benchmark(query, k=k, metric=metric)
        return jsonify({
            "bruteforceUs": result["bruteforce"],
            "kdtreeUs": result["kdtree"],
            "hnswUs": result["hnsw"],
            "itemCount": len(db.items),
        })
    except (ValueError, TypeError) as exc:
        return error_response(exc)


@app.get("/hnsw-info")
def hnsw_info():
    # USearch does not expose the complete adjacency graph through
    # the API currently used here; never fabricate edge data.
    nodes = [
        {
            "id": item["id"],
            "metadata": item["metadata"],
            "category": item["category"],
            "maxLyr": 0,
        }
        for item in db.items.values()
    ]
    return jsonify({
        "topLayer": 0,
        "nodeCount": len(nodes),
        "nodesPerLayer": [len(nodes)],
        "edgesPerLayer": [None],
        "nodes": nodes,
        "edges": [],
        "graphNote": (
            "Node metadata is available; actual HNSW adjacency edges "
            "are not exposed by this implementation."
        ),
    })


@app.get("/status")
def status():
    return jsonify({
        "ollamaAvailable": ollama.is_available(),
        "embedModel": ollama.embed_model,
        "genModel": ollama.gen_model,
        "docDims": doc_db.dims or 0,
        "docCount": doc_db.count,
    })


@app.get("/stats")
def stats():
    return jsonify({
        "count": len(db.items),
        "dims": db.dims,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"],
        "docCount": doc_db.count,
    })


@app.post("/doc/insert")
def insert_document():
    try:
        data = parse_json_body()
        title = str(data.get("title", "")).strip()
        text = str(data.get("text", "")).strip()
        if not title or not text:
            raise ValueError("Both title and text are required")
        if len(text) > 500_000:
            raise ValueError("Document text exceeds the 500,000 character limit")

        chunks = doc_db.chunk_text(text, chunk_words=250, overlap_words=30)
        if not chunks:
            raise ValueError("No text to index")

        # Embed all chunks first, avoiding partial storage on embedding errors.
        embeddings = [embed_or_raise(chunk) for chunk in chunks]
        ids = []
        try:
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings), 1):
                chunk_title = f"{title} [{i}/{len(chunks)}]" if len(chunks) > 1 else title
                ids.append(doc_db.insert(chunk_title, chunk, embedding))
        except Exception:
            for inserted_id in ids:
                doc_db.delete(inserted_id)
            raise

        return jsonify({"ids": ids, "chunks": len(ids), "dims": doc_db.dims}), 201
    except RuntimeError as exc:
        return error_response(exc, 503)
    except (ValueError, TypeError) as exc:
        return error_response(exc)
    except Exception:
        app.logger.exception("Document insertion failed")
        return error_response("Could not insert document.", 500)


@app.delete("/doc/delete/<int:doc_id>")
def delete_document(doc_id):
    try:
        return jsonify({"ok": doc_db.delete(doc_id)})
    except Exception:
        app.logger.exception("Document deletion failed")
        return error_response("Could not delete document.", 500)


@app.get("/doc/list")
def list_documents():
    return jsonify(doc_db.list_documents())


@app.post("/doc/search")
def search_documents():
    try:
        data = parse_json_body()
        question = str(data.get("question", "")).strip()
        k = parse_k(data.get("k", 3))
        if not question:
            raise ValueError("question is required")
        if doc_db.count == 0:
            return jsonify({"contexts": []})
        hits = doc_db.search(embed_or_raise(question), k)
        return jsonify({
            "contexts": [
                {"id": hit["id"], "title": hit["title"],
                 "distance": round(hit["distance"], 6)}
                for hit in hits
            ]
        })
    except RuntimeError as exc:
        return error_response(exc, 503)
    except (ValueError, TypeError) as exc:
        return error_response(exc)
    except Exception:
        app.logger.exception("Document search failed")
        return error_response("Could not search documents.", 500)


@app.post("/doc/ask")
def ask_documents():
    try:
        data = parse_json_body()
        question = str(data.get("question", "")).strip()
        k = parse_k(data.get("k", 3))
        if not question:
            raise ValueError("question is required")
        if doc_db.count == 0:
            return error_response("No documents indexed yet. Insert a document first.", 400)

        hits = doc_db.search(embed_or_raise(question), k)
        context = "\n\n".join(
            f"[{i}] {hit['title']}:\n{hit['text']}"
            for i, hit in enumerate(hits, 1)
        )
        prompt = (
            "Answer the user's question using the retrieved material below "
            "when relevant. Do not invent facts attributed to these documents. "
            "If the material does not answer the question, say so clearly and "
            "distinguish general knowledge from document facts.\n\n"
            f"Retrieved material:\n{context or '(No relevant chunks found)'}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        answer = ollama.generate(prompt)
        if not answer:
            return error_response(
                "Could not generate an answer. Check Ollama and run: ollama pull llama3.2",
                503,
            )
        return jsonify({
            "answer": answer,
            "model": ollama.gen_model,
            "contexts": [
                {"id": hit["id"], "title": hit["title"], "text": hit["text"],
                 "distance": round(hit["distance"], 6)}
                for hit in hits
            ],
            "docCount": doc_db.count,
        })
    except RuntimeError as exc:
        return error_response(exc, 503)
    except (ValueError, TypeError) as exc:
        return error_response(exc)
    except Exception:
        app.logger.exception("RAG generation failed")
        return error_response("Could not generate an answer.", 500)


if __name__ == "__main__":
    print("VectorNova: http://127.0.0.1:8080")
    print(f"Loaded {len(db.items)} vectors ({db.dims} dimensions)")
    print(f"Loaded {doc_db.count} document chunks from SQLite")
    app.run(host="127.0.0.1", port=8080, debug=False)
