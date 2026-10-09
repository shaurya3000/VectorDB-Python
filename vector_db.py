
import time
import numpy as np
from sklearn.neighbors import KDTree
from usearch.index import Index


def distance(a, b, metric="cosine"):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if metric == "euclidean":
        return float(np.linalg.norm(a - b))

    if metric == "manhattan":
        return float(np.abs(a - b).sum())

    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator < 1e-9:
        return 1.0

    return float(1 - np.dot(a, b) / denominator)


class VectorDB:
    def __init__(self, dims=16):
        self.dims = dims
        self.items = {}
        self.next_id = 1

        self.kdtree = None
        self.kd_ids = []

        # Real HNSW index using cosine similarity.
        self.hnsw = Index(
            ndim=dims,
            metric="cos",
            connectivity=16,
            expansion_add=128,
            expansion_search=64,
            dtype="f32"
        )

    def insert(self, metadata, category, embedding):
        embedding = np.asarray(embedding, dtype=np.float32)

        if embedding.shape != (self.dims,):
            raise ValueError(
                f"Expected a vector with {self.dims} dimensions"
            )

        if not np.all(np.isfinite(embedding)):
            raise ValueError("Embedding contains invalid values")

        item_id = self.next_id
        self.next_id += 1

        item = {
            "id": item_id,
            "metadata": str(metadata),
            "category": str(category),
            "embedding": embedding.tolist()
        }

        self.items[item_id] = item

        try:
            self.hnsw.add(item_id, embedding)
        except Exception:
            del self.items[item_id]
            self.next_id -= 1
            raise

        self._rebuild_kdtree()
        return item_id

    def _rebuild_kdtree(self):
        self.kd_ids = list(self.items.keys())

        if self.kd_ids:
            vectors = [
                self.items[item_id]["embedding"]
                for item_id in self.kd_ids
            ]
            self.kdtree = KDTree(np.asarray(vectors))
        else:
            self.kdtree = None

    def delete(self, item_id):
        if item_id not in self.items:
            return False

        # Remove from HNSW as well as the item store.
        self.hnsw.remove(item_id)
        del self.items[item_id]

        self._rebuild_kdtree()
        return True

    def search(
        self,
        query,
        k=5,
        metric="cosine",
        algorithm="bruteforce"
    ):
        query = np.asarray(query, dtype=np.float32)

        if query.shape != (self.dims,):
            raise ValueError(
                f"Query must have {self.dims} dimensions"
            )

        if not np.all(np.isfinite(query)):
            raise ValueError("Query contains invalid values")

        if k < 1:
            raise ValueError("k must be at least 1")

        if metric not in ("cosine", "euclidean", "manhattan"):
            raise ValueError("Unsupported distance metric")

        if algorithm not in ("hnsw", "kdtree", "bruteforce"):
            raise ValueError("Unsupported search algorithm")

        start = time.perf_counter()

        if not self.items:
            return {
                "hits": [],
                "latency_us": 0.0,
                "algorithm": algorithm,
                "metric": metric
            }

        count = min(k, len(self.items))
        candidates = []

        if algorithm == "hnsw" and metric == "cosine":
            # USearch returns a Matches object; its keys are vector IDs.
            matches = self.hnsw.search(query, count)
            keys = matches.keys if hasattr(matches, "keys") else matches

            candidates = [
                (
                    distance(
                        query,
                        self.items[int(item_id)]["embedding"],
                        "cosine"
                    ),
                    int(item_id)
                )
                for item_id in keys
                if int(item_id) in self.items
            ]

            candidates.sort(key=lambda pair: pair[0])

        elif algorithm == "kdtree" and metric == "euclidean":
            distances, indices = self.kdtree.query(
                query.reshape(1, -1),
                k=count
            )

            candidates = [
                (float(dist), self.kd_ids[int(index)])
                for dist, index in zip(
                    distances[0], indices[0]
                )
            ]

        else:
            # Exact ranking for Brute Force and for metrics
            # other than cosine when HNSW is selected.
            candidates = [
                (
                    distance(query, item["embedding"], metric),
                    item_id
                )
                for item_id, item in self.items.items()
            ]

            candidates.sort(key=lambda pair: pair[0])
            candidates = candidates[:count]

        elapsed = (time.perf_counter() - start) * 1_000_000

        hits = []
        for dist, item_id in candidates[:count]:
            hits.append({
                **self.items[item_id],
                "distance": float(dist)
            })

        return {
            "hits": hits,
            "latency_us": round(elapsed, 2),
            "algorithm": algorithm,
            "metric": metric
        }

    def benchmark(self, query, k=5, metric="cosine"):
        results = {}

        for algorithm in ("bruteforce", "kdtree", "hnsw"):
            result = self.search(
                query,
                k=k,
                metric=metric,
                algorithm=algorithm
            )
            results[algorithm] = result["latency_us"]

        return results


if __name__ == "__main__":
    db = VectorDB(dims=4)

    db.insert("A", "cs", [1, 0, 0, 0])
    db.insert("B", "math", [0, 1, 0, 0])
    db.insert("C", "cs", [0.9, 0.1, 0, 0])

    query = [1, 0, 0, 0]

    for algorithm in ("bruteforce", "kdtree", "hnsw"):
        result = db.search(
            query,
            k=2,
            metric="cosine",
            algorithm=algorithm
        )

        print(f"\n{algorithm.upper()}")
        print("Latency:", result["latency_us"], "microseconds")

        for hit in result["hits"]:
            print(
                hit["id"],
                hit["metadata"],
                "distance:",
                round(hit["distance"], 5)
            )

    print("\nBenchmark:", db.benchmark(query))
