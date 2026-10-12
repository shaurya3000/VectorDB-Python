import time
import numpy as np
from usearch.index import Index

N_VECTORS = 10_000
DIMS = 16
N_QUERIES = 100
K = 5
WARMUP_QUERIES = 10
SEED = 42


def normalize_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def main():
    rng = np.random.default_rng(SEED)
    vectors = normalize_rows(rng.normal(size=(N_VECTORS, DIMS)).astype(np.float32))
    queries = normalize_rows(rng.normal(size=(N_QUERIES, DIMS)).astype(np.float32))

    print(f"Building HNSW index: {N_VECTORS:,} vectors × {DIMS} dimensions...")
    index = Index(
        ndim=DIMS, metric="cos", connectivity=16,
        expansion_add=128, expansion_search=64, dtype="f32"
    )
    ids = np.arange(1, N_VECTORS + 1, dtype=np.int64)
    t0 = time.perf_counter()
    index.add(ids, vectors)
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"Index build time: {build_ms:.2f} ms")

    for q in queries[:WARMUP_QUERIES]:
        index.search(q, K)
        distances = 1.0 - vectors @ q
        np.argpartition(distances, K - 1)[:K]

    hnsw_times, exact_times, recalls = [], [], []
    print(f"Running {N_QUERIES} queries (k={K})...")
    for q in queries:
        t0 = time.perf_counter_ns()
        matches = index.search(q, K)
        hnsw_times.append((time.perf_counter_ns() - t0) / 1000.0)
        hnsw_ids = set(int(v) for v in matches.keys)

        t0 = time.perf_counter_ns()
        # Exact cosine distance for normalized vectors is 1 - dot product.
        distances = 1.0 - vectors @ q
        exact_indices = np.argpartition(distances, K - 1)[:K]
        exact_indices = exact_indices[np.argsort(distances[exact_indices])]
        exact_times.append((time.perf_counter_ns() - t0) / 1000.0)
        exact_ids = set(int(i + 1) for i in exact_indices)
        recalls.append(len(hnsw_ids & exact_ids) / K)

    def report(label, values):
        values = np.asarray(values, dtype=np.float64)
        print(
            f"{label:<22} mean={np.mean(values):9.2f} µs  "
            f"median={np.median(values):9.2f} µs  "
            f"p95={np.percentile(values, 95):9.2f} µs"
        )

    print("\n=== VectorNova HNSW Benchmark ===")
    print(f"Vectors: {N_VECTORS:,} | Dimensions: {DIMS} | Queries: {N_QUERIES} | k: {K}")
    report("HNSW", hnsw_times)
    report("Brute-force exact", exact_times)
    print(f"Recall@{K}: {np.mean(recalls) * 100:.2f}%")
    print(f"Index build time: {build_ms:.2f} ms")
    print("\nNotes:")
    print("- Search latency excludes index build time.")
    print("- Results vary with hardware, background load, and library versions.")
    print("- This uses VectorNova's HNSW settings on an isolated synthetic dataset.")
    print("- Exact search uses NumPy vectorized cosine distance and top-k selection.")


if __name__ == "__main__":
    main()
