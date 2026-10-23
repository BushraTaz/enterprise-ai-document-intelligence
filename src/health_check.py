
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "volvo_annual_report"


def check_qdrant_connection(client):
    try:
        client.get_collections()
        print("[OK] Qdrant is reachable.")
        return True
    except Exception as exc:
        print("[ERROR] Cannot connect to Qdrant.")
        print("Check that your existing Qdrant container is running.")
        print(f"Details: {exc}")
        return False


def check_collection(client):
    try:
        collection = client.get_collection(COLLECTION_NAME)
        point_count = collection.points_count

        print(f"[OK] Collection '{COLLECTION_NAME}' exists.")
        print(f"[INFO] Stored points: {point_count}")

        if not point_count:
            print("[WARNING] The collection is empty.")
            print("Check your document-ingestion process.")

        return point_count > 0

    except UnexpectedResponse as exc:
        print(f"[ERROR] Could not inspect '{COLLECTION_NAME}'.")
        print("Check that the collection exists and Qdrant is healthy.")
        print(f"Details: {exc}")
        return False
    except Exception as exc:
        print(f"[ERROR] Could not inspect '{COLLECTION_NAME}'.")
        print(f"Details: {exc}")
        return False


def main():
    client = QdrantClient(url=QDRANT_URL, timeout=5)

    if not check_qdrant_connection(client):
        return

    if not check_collection(client):
        print("\nHealth check failed. No data was changed.")
        return

    print("\nHealth check passed. Your document collection is ready.")


if __name__ == "__main__":
    main()
