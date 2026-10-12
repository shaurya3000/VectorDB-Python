import json
import sqlite3
from pathlib import Path

import numpy as np
from usearch.index import Index

DB_PATH = Path(__file__).resolve().parent / "vector_nova.db"


class DocumentDB:
    def __init__(self, db_path=DB_PATH):
        self.db_path = str(db_path)
        self.items = {}
        self.dims = None
        self.index = None

        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL
                )
            """)
        self._load()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    @staticmethod
    def chunk_text(text, chunk_words=250, overlap_words=30):
        if chunk_words < 1 or not 0 <= overlap_words < chunk_words:
            raise ValueError("Invalid chunk_words or overlap_words.")
        words = text.split()
        if not words:
            return []
        step = chunk_words - overlap_words
        return [" ".join(words[i:i + chunk_words])
                for i in range(0, len(words), step)]

    def _make_index(self, dims):
        return Index(
            ndim=dims, metric="cos", connectivity=16,
            expansion_add=128, expansion_search=64, dtype="f32"
        )

    def _load(self):
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT id, title, text, embedding FROM documents ORDER BY id
            """).fetchall()
        if not rows:
            return

        parsed = []
        for item_id, title, text, embedding_json in rows:
            vector = np.asarray(json.loads(embedding_json), dtype=np.float32)
            if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
                raise ValueError(f"Invalid stored embedding for document {item_id}.")
            parsed.append((int(item_id), title, text, vector))

        dims = len(parsed[0][3])
        if any(len(vector) != dims for _, _, _, vector in parsed):
            raise ValueError("Stored document embeddings have inconsistent dimensions.")

        self.dims = dims
        self.index = self._make_index(dims)
        for item_id, title, text, vector in parsed:
            self.index.add(item_id, vector)
            self.items[item_id] = {
                "id": item_id, "title": title, "text": text,
                "embedding": vector.tolist()
            }

    def insert(self, title, text, embedding):
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("Embedding must be a non-empty 1D vector.")
        if not np.isfinite(vector).all():
            raise ValueError("Embedding contains non-finite values.")
        if not str(title).strip():
            raise ValueError("Title cannot be empty.")
        if self.dims is not None and len(vector) != self.dims:
            raise ValueError("Embedding dimensions do not match existing documents.")

        if self.index is None:
            new_index = self._make_index(len(vector))
        else:
            new_index = self.index

        # Persist first. The AUTOINCREMENT id is the stable HNSW ID.
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO documents (title, text, embedding) VALUES (?, ?, ?)",
                (str(title), str(text), json.dumps(vector.tolist()))
            )
            item_id = int(cursor.lastrowid)

        try:
            if self.index is None:
                self.dims = len(vector)
                self.index = new_index
            self.index.add(item_id, vector)
        except Exception:
            with self._connect() as conn:
                conn.execute("DELETE FROM documents WHERE id = ?", (item_id,))
            raise

        self.items[item_id] = {
            "id": item_id, "title": str(title), "text": str(text),
            "embedding": vector.tolist()
        }
        return item_id

    def search(self, embedding, k=3):
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        if not self.items or self.index is None:
            return []

        query = np.asarray(embedding, dtype=np.float32)
        if query.ndim != 1 or len(query) != self.dims:
            raise ValueError("Query embedding has incorrect dimensions.")
        if not np.isfinite(query).all():
            raise ValueError("Query embedding contains non-finite values.")

        matches = self.index.search(query, min(k, len(self.items)))
        ids = matches.keys if hasattr(matches, "keys") else matches
        query_norm = float(np.linalg.norm(query))
        results = []
        for raw_id in ids:
            item_id = int(raw_id)
            item = self.items.get(item_id)
            if item is None:
                continue
            vector = np.asarray(item["embedding"], dtype=np.float32)
            denominator = query_norm * float(np.linalg.norm(vector))
            similarity = float(np.dot(query, vector) / denominator) if denominator else 0.0
            results.append({
                **item,
                "distance": float(1.0 - np.clip(similarity, -1.0, 1.0))
            })
        return sorted(results, key=lambda item: item["distance"])[:k]

    def delete(self, item_id):
        item_id = int(item_id)
        if item_id not in self.items:
            return False
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (item_id,))
        if self.index is not None:
            self.index.remove(item_id)
        del self.items[item_id]
        return True

    def list_documents(self):
        return [
            {
                "id": item["id"],
                "title": item["title"],
                "preview": item["text"][:120],
                "words": len(item["text"].split()),
            }
            for item in self.items.values()
        ]

    @property
    def count(self):
        return len(self.items)
