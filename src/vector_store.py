
from pathlib import Path
import hashlib

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "artifacts" / "chroma_db"
COLLECTION_NAME = "knowledge_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self):
        DB_PATH.mkdir(parents=True, exist_ok=True)

        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=str(DB_PATH))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def make_document_id(filename: str, file_bytes: bytes) -> str:
        content_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
        safe_name = Path(filename).name
        return f"{safe_name}__{content_hash}"

    def add_chunks(
        self,
        chunks: list[dict],
        document_id: str | None = None,
    ) -> int:
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for index, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            if not text:
                continue

            source = str(chunk.get("source", "unknown"))
            page = int(chunk.get("page", 0))
            start_index = int(chunk.get("start_index", 0))

            unique_id = (
                f"{document_id or source}::page-{page}"
                f"::start-{start_index}::chunk-{index}"
            )

            ids.append(unique_id)
            documents.append(text)
            metadatas.append({
                "source": source,
                "page": page,
                "start_index": start_index,
                "document_id": document_id or source,
            })

        if not documents:
            return 0

        embeddings = self.embedder.encode(
            documents,
            normalize_embeddings=True,
        ).tolist()

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        return len(documents)

    def list_documents(self) -> list[dict]:
        stored = self.collection.get(include=["metadatas"])
        documents = {}

        for metadata in stored.get("metadatas") or []:
            if not metadata:
                continue

            doc_id = metadata.get(
                "document_id",
                metadata.get("source", "unknown"),
            )
            documents.setdefault(
                doc_id,
                metadata.get("source", "unknown"),
            )

        return [
            {"document_id": doc_id, "source": source}
            for doc_id, source in sorted(documents.items())
        ]

    def search(
        self,
        query: str,
        top_k: int = 4,
        document_id: str | None = None,
    ) -> list[dict]:
        query = query.strip()

        if not query:
            return []

        # count() is unfiltered; apply document filtering in query().
        total_count = self.collection.count()

        if total_count == 0:
            return []

        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True,
        )[0].tolist()

        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, total_count),
            "include": ["documents", "metadatas", "distances"],
        }

        if document_id:
            query_args["where"] = {"document_id": document_id}

        results = self.collection.query(**query_args)

        matches = []
        result_documents = results.get("documents") or [[]]

        if not result_documents or not result_documents[0]:
            return []

        for index, document in enumerate(result_documents[0]):
            metadata = results["metadatas"][0][index]

            matches.append({
                "text": document,
                "source": metadata["source"],
                "page": metadata["page"],
                "document_id": metadata.get(
                    "document_id",
                    metadata["source"],
                ),
                "distance": results["distances"][0][index],
            })

        return matches


