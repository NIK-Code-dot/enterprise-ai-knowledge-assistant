
from src.chunking import split_pages
from src.vector_store import VectorStore


def main():
    sample_pages = [
        {
            "source": "sample_handbook.txt",
            "page": 1,
            "text": (
                "The company provides 20 paid vacation days annually. "
                "Employees should submit leave requests to their manager. "
                "The IT help desk operates from 9 AM to 6 PM on weekdays."
            ),
        }
    ]

    chunks = split_pages(sample_pages)
    print("Chunks created:", len(chunks))

    store = VectorStore()
    added = store.add_chunks(chunks)
    print("Chunks indexed:", added)

    results = store.search("What is the industrial dispute and strike case about?")

    print("\nSearch results:")
    for result in results:
        print(f"\nSource: {result['source']} | Page: {result['page']}")
        print(result["text"])
        print("Distance:", round(result["distance"], 4))


if __name__ == "__main__":
    main()

