from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "volvo_annual_report"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
QDRANT_URL = "http://localhost:6333"


def retrieve(client, model, query, limit=10):
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
    ).points


def main():
    client = QdrantClient(url=QDRANT_URL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    question = input("Ask a question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    # Add targeted wording for employee-count questions.
    queries = [question]

    if any(word in question.lower() for word in (
        "employee", "employees", "workforce", "staff"
    )):
        queries.append(
            "Volvo Group number of employees in 2025 and 2024 "
            "89,188 90,995 employee count"
        )

    # Merge results, keeping the best score for each point.
    merged = {}

    for query in queries:
        for result in retrieve(client, model, query, limit=10):
            key = str(result.id)

            if key not in merged or result.score > merged[key].score:
                merged[key] = result

    ranked = sorted(
        merged.values(),
        key=lambda result: result.score,
        reverse=True,
    )[:10]

    print("\nMerged retrieval results:\n")

    for i, result in enumerate(ranked, start=1):
        payload = result.payload or {}

        print(f"--- Result {i} ---")
        print(f"Score: {result.score:.4f}")
        print(f"Page: {payload.get('page', 'unknown')}")
        print(f"Source: {payload.get('source', 'unknown')}")
        print((payload.get("text") or "")[:1000])
        print()


if __name__ == "__main__":
    main()
