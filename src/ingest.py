from pathlib import Path
from pypdf import PdfReader


PDF_PATH = Path("data/5313699-volvo-group---annual-report-2025.pdf")


def load_pdf(path: Path):
    reader = PdfReader(path)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        documents.append({
            "page": page_number,
            "text": text
        })

    return documents


if __name__ == "__main__":
    pages = load_pdf(PDF_PATH)

    print(f"Loaded {len(pages)} pages")

    for page in pages[:3]:
        print("\n--- PAGE", page["page"], "---")
        print(page["text"][:1000])