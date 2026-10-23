from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from chunker import create_chunks
from ingest import load_pdf, PDF_PATH


COLLECTION_NAME = "volvo_annual_report"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def main():
    # 1. Load and chunk the PDF
    pages = load_pdf(PDF_PATH)
    chunks = create_chunks(pages)

    print(f"Loaded {len(chunks)} chunks")

    # 2. Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # 3. Create embeddings
    print("Creating embeddings...")
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print(f"Created {len(embeddings)} embeddings")
    print(f"Vector dimension: {len(embeddings[0])}")

    # 4. Connect to Qdrant
    client = QdrantClient(url="http://localhost:6333")

    # 5. Create collection
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=len(embeddings[0]),
            distance=Distance.COSINE
        )
    )

    # 6. Prepare points
    points = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={
                    "text": chunk["text"],
                    "page": chunk["page"],
                    "chunk_index": chunk["chunk_index"],
                    "source": chunk["source"],
                }
            )
        )

    # 7. Store vectors
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"Stored {len(points)} vectors in Qdrant")
    print("Done!")


if __name__ == "__main__":
    main()