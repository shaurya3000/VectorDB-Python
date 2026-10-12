import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "vector_nova.db"


class VectorStorage:
    def __init__(self, db_path=DB_PATH):
        self.db_path = str(db_path)
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY,
                    metadata TEXT NOT NULL,
                    category TEXT NOT NULL,
                    embedding TEXT NOT NULL
                )
            """)

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def load(self):
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT id, metadata, category, embedding
                FROM vectors ORDER BY id
            """).fetchall()
        return [
            {
                "id": int(row[0]),
                "metadata": row[1],
                "category": row[2],
                "embedding": json.loads(row[3]),
            }
            for row in rows
        ]

    def save(self, item_id, metadata, category, embedding):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO vectors (id, metadata, category, embedding)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    metadata=excluded.metadata,
                    category=excluded.category,
                    embedding=excluded.embedding
            """, (
                int(item_id), str(metadata), str(category),
                json.dumps([float(x) for x in embedding]),
            ))

    def delete(self, item_id):
        with self._connect() as conn:
            conn.execute("DELETE FROM vectors WHERE id = ?", (int(item_id),))
