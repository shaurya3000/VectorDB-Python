"""
VectorNova: HNSW parameter comparison at 100,000 vectors.

Isolated benchmark only: does not modify app.py, vector_db.py, or SQLite data.
Compares expansion_search values 64 and 128 against exact cosine top-k.
"""
import time
import numpy as np
from usearch.index import Index

N_VECTORS = 100_000
DIMS = 16
N_QUERIES = 100
K_VALUES = (5, 10)
EXPANSIONS = (64, 128)
SEED = 42


def normalize_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def exact_topk(vectors, query, k):
    distances = 1.0 - vectors @ query
    ids = np.argpartition(distances, k - 1)[:k]
    ids = ids[np.argsort(distances[ids])]
    return ids + 1


def stats(label, values):
    values = np.asarray(values, dtype=np.float64)
    print(
        f"{label:<25} mean={np.mean(values):8.2f} µs  "
        f"median={np.median(values):8.2f} µs  "
        f"p95={np.percentile(values, 95):8.2f} µs"
    )


def main():
    rng = np.random.default_rng(SEED)
    vectors = normalize_rows(rng.normal(size=(N_VECTORS, DIMS)).astype(np.float32))
    queries = normalize_rows(rng.normal(size=(N_QUERIES, DIMS)).astype(np.float32))

    print(f"Dataset: {N_VECTORS:,} vectors × {DIMS} dimensions")
    print(f"Queries: {N_QUERIES} | k values: {K_VALUES}")
    exact = {k: [] for k in K_VALUES}
    exact_times = []
    for q in queries:
        t0 = time.perf_counter_ns()
        results = {k: exact_topk(vectors, q, k) for k in K_VALUES}
        exact_times.append((time.perf_counter_ns() - t0) / 1000.0)
        for k in K_VALUES:
            exact[k].append(set(int(x) for x in results[k]))

    stats("Exact brute-force top-k", exact_times)
    print()

    for expansion in EXPANSIONS:
        index = Index(
            ndim=DIMS, metric="cos", connectivity=16,
            expansion_add=128, expansion_search=expansion, dtype="f32"
        )
        ids = np.arange(1, N_VECTORS + 1, dtype=np.int64)
        t0 = time.perf_counter()
        index.add(ids, vectors)
        build_ms = (time.perf_counter() - t0) * 1000.0

        # Warm-up separate from timed queries.
        for q in queries[:10]:
            index.search(q, max(K_VALUES))

        latencies = []
        recall = {k: [] for k in K_VALUES}
        for q in queries:
            t0 = time.perf_counter_ns()
            matches = index.search(q, max(K_VALUES))
            latencies.append((time.perf_counter_ns() - t0) / 1000.0)
            found = [int(x) for x in matches.keys]
            found_sets = {k: set(found[:k]) for k in K_VALUES}
            for k in K_VALUES:
                recall[k].append(len(found_sets[k] & exact[k][len(latencies) - 1]) / k)

        print(f"=== expansion_search={expansion} ===")
        stats("HNSW search", latencies)
        print(f"Index build: {build_ms:.2f} ms")
        for k in K_VALUES:
            print(f"Recall@{k}: {np.mean(recall[k]) * 100:.2f}%")
        print()

    print("Notes: each expansion setting gets a separately built index.")
    print("Synthetic normalized vectors; hardware and library versions affect timings.")
    print("Recall compares HNSW's returned top-k IDs against exact cosine top-k.")


if __name__ == "__main__":
    main()
