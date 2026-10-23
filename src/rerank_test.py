from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

COLLECTION_NAME = "volvo_annual_report"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CANDIDATE_K = 10
FINAL_K = 5


def retrieve_candidates(question, embedding_model, client):
    query_vector = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=CANDIDATE_K,
        with_payload=True
    ).points

    return results


def rerank(question, results, reranker):
    pairs = [
        (question, result.payload["text"])
        for result in results
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(results, scores),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked


def main():
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading reranker...")
    reranker = CrossEncoder(RERANKER_MODEL)

    client = QdrantClient(url="http://localhost:6333")

    question = input("\nAsk a question: ")

    print("\nRetrieving top 10 candidates...")
    candidates = retrieve_candidates(
        question,
        embedding_model,
        client
    )

    print("\nReranking candidates...")
    ranked = rerank(question, candidates, reranker)

    print("\n=== RERANKED RESULTS ===\n")

    for rank, (result, score) in enumerate(
        ranked[:FINAL_K],
        start=1
    ):
        payload = result.payload

        print(f"--- Rank {rank} ---")
        print(f"Reranker score: {score:.4f}")
        print(f"Original Qdrant score: {result.score:.4f}")
        print(f"Page: {payload['page']}")
        print(f"Source: {payload['source']}")
        print()
        print(payload["text"][:700])
        print()


if __name__ == "__main__":
    main()