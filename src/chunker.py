from ingest import load_pdf, PDF_PATH


CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping word-based chunks.

    Example:
        chunk 1 = words 0-800
        chunk 2 = words 650-1450
        chunk 3 = words 1300-2100
    """

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_chunks(pages):
    """
    Convert PDF pages into chunks while preserving page metadata.
    """

    chunks = []

    for page in pages:

        page_chunks = chunk_text(page["text"])

        for index, chunk in enumerate(page_chunks):

            chunks.append({
                "text": chunk,
                "page": page["page"],
                "chunk_index": index,
                "source": PDF_PATH.name
            })

    return chunks


if __name__ == "__main__":

    pages = load_pdf(PDF_PATH)

    chunks = create_chunks(pages)

    print(f"Pages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")

    for chunk in chunks[:3]:
        print("\n--- CHUNK ---")
        print("Source:", chunk["source"])
        print("Page:", chunk["page"])
        print("Chunk:", chunk["chunk_index"])
        print(chunk["text"][:1000])