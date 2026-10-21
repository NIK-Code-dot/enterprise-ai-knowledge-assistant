
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_pages(
    pages: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[dict]:
    """Split extracted PDF pages into searchable text chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    chunks = []

    for page in pages:
        text = page["text"]

        for chunk in splitter.create_documents(
            [text],
            metadatas=[{
                "source": page["source"],
                "page": page["page"],
            }],
        ):
            chunks.append({
                "text": chunk.page_content,
                "source": chunk.metadata["source"],
                "page": chunk.metadata["page"],
                "start_index": chunk.metadata["start_index"],
            })

    return chunks


if __name__ == "__main__":
    sample_pages = [{
        "source": "demo.pdf",
        "page": 1,
        "text": "Enterprise AI systems retrieve relevant information from documents. " * 40,
    }]

    result = split_pages(sample_pages)

    print("Chunking working:", bool(result))
    print("Total chunks:", len(result))
    print("First chunk:", result[0]["text"][:150])
    print("Source:", result[0]["source"])
    print("Page:", result[0]["page"])
