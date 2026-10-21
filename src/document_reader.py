
from pathlib import Path
from pypdf import PdfReader


def extract_pdf_text(pdf_path: str) -> list[dict]:
    """Extract text from a PDF, preserving page numbers."""

    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        if text:
            pages.append({
                "page": page_number,
                "text": text,
                "source": path.name,
            })

    return pages



if __name__ == "__main__":
    pdf_path = input("Enter PDF path: ").strip().strip('"')

    try:
        pages = extract_pdf_text(pdf_path)

        print(f"\nPages with extracted text: {len(pages)}")

        for item in pages[:3]:
            print(f"\n--- Page {item['page']} ---")
            print(item["text"][:500])

        if not pages:
            print("No extractable text found. This PDF may need OCR.")

    except Exception as error:
        print(f"PDF test failed: {error}")
