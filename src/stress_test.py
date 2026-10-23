
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from math import ceil

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "volvo_annual_report"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Small, controlled test: 20 searches at each concurrency level.
REQUESTS_PER_LEVEL = 20
CONCURRENCY_LEVELS = [1, 2, 4]

QUESTIONS = [
    "What were Volvo Group's net sales in 2025?",
    "What was the group's operating income in 2025?",
    "What were the main financial risks?",
    "How did truck sales perform?",
    "What are the group's sustainability goals?",
]

thread_local = threading.local()


def get_client():
    # Reuse one Qdrant client per worker thread.
    if not hasattr(thread_local, "client"):
        thread_local.client = QdrantClient(url=QDRANT_URL)
    return thread_local.client


def percentile(values, p):
    ordered = sorted(values)
    return ordered[max(0, ceil(p * len(ordered)) - 1)]


def search_once(vector):
    client = get_client()
    start = time.perf_counter()
    try:
        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=5,
            with_payload=True,
        )
        elapsed = time.perf_counter() - start
        return elapsed, len(result.points), None
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return elapsed, 0, str(exc)


def run_level(vectors, concurrency):
    jobs = [
        vectors[i % len(vectors)]
        for i in range(REQUESTS_PER_LEVEL)
    ]

    durations = []
    errors = []
    total_results = 0

    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(search_once, vector) for vector in jobs]

        for future in as_completed(futures):
            elapsed, count, error = future.result()
            durations.append(elapsed)
            total_results += count
            if error:
                errors.append(error)

    wall_time = time.perf_counter() - wall_start
    successful = len(durations) - len(errors)

    print(f"\nConcurrency: {concurrency}")
    print(f"Requests:    {len(jobs)}")
    print(f"Successful:  {successful}")
    print(f"Errors:      {len(errors)}")
    print(f"Results returned: {total_results}")
    print(f"Wall time:   {wall_time:.3f} seconds")
    print(f"Throughput:  {successful / wall_time:.2f} successful searches/sec")

    if durations:
        print(f"Mean latency: {mean(durations) * 1000:.2f} ms")
        print(f"Median/p50:   {percentile(durations, 0.50) * 1000:.2f} ms")
        print(f"p95 latency:  {percentile(durations, 0.95) * 1000:.2f} ms")
        print(f"Max latency:  {max(durations) * 1000:.2f} ms")

    if errors:
        print("First error:", errors[0])


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Checking Qdrant collection...")
    client = QdrantClient(url=QDRANT_URL)
    info = client.get_collection(COLLECTION_NAME)
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored points: {info.points_count}")

    print("\nPreparing test queries...")
    vectors = [
        model.encode(question, normalize_embeddings=True).tolist()
        for question in QUESTIONS
    ]

    # One warm-up search, excluded from benchmark measurements.
    warmup = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vectors[0],
        limit=5,
        with_payload=True,
    )
    print(f"Warm-up successful; returned {len(warmup.points)} results.")

    print("\nStarting Qdrant retrieval benchmark...")
    for concurrency in CONCURRENCY_LEVELS:
        run_level(vectors, concurrency)

    print("\nBenchmark complete. No collection changes were made.")


if __name__ == "__main__":
    main()