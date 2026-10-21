
from pathlib import Path

from src.document_reader import extract_pdf_text
from src.chunking import split_pages
from src.vector_store import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


def main():
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/documents.")
        return

    store = VectorStore()

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")

        try:
            file_bytes = pdf_path.read_bytes()
            document_id = store.make_document_id(
                pdf_path.name,
                file_bytes,
            )

            pages = extract_pdf_text(str(pdf_path))
            chunks = split_pages(pages)

            if not chunks:
                print("No extractable text found. PDF may need OCR.")
                continue

            indexed = store.add_chunks(
                chunks,
                document_id=document_id,
            )

            print(f"Document ID: {document_id}")
            print(f"Pages with text: {len(pages)}")
            print(f"Chunks created: {len(chunks)}")
            print(f"Chunks indexed: {indexed}")

        except Exception as error:
            print(f"Failed to process {pdf_path.name}: {error}")

    print("\nDocument ingestion complete!")


if __name__ == "__main__":
    main()

