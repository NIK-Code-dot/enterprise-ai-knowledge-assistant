
from pathlib import Path
import hashlib
import math
import os

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "artifacts" / "chroma_db"
COLLECTION_NAME = "knowledge_documents"

# Local development keeps the existing embedding model.
# Set EMBEDDING_BACKEND=fastembed on Render to reduce memory usage.
EMBEDDING_BACKEND = os.getenv(
    "EMBEDDING_BACKEND",
    "sentence-transformers",
).strip().lower()

SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"


class VectorStore:
    def __init__(self):
        DB_PATH.mkdir(parents=True, exist_ok=True)

        # Avoid loading the embedding model during API startup.
        self._embedder = None

        self.client = chromadb.PersistentClient(path=str(DB_PATH))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _get_embedder(self):
        """Load the selected embedding model only when first needed."""
        if self._embedder is not None:
            return self._embedder

        if EMBEDDING_BACKEND == "fastembed":
            try:
                from fastembed import TextEmbedding
            except ImportError as error:
                raise RuntimeError(
                    "FastEmbed is selected but not installed. "
                    "Add fastembed to requirements.txt and redeploy."
                ) from error

            self._embedder = TextEmbedding(
                model_name=FASTEMBED_MODEL
            )

        elif EMBEDDING_BACKEND == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "sentence-transformers is not installed."
                ) from error

            self._embedder = SentenceTransformer(
                SENTENCE_TRANSFORMER_MODEL,
                device="cpu",
            )

        else:
            raise ValueError(
                "Unsupported EMBEDDING_BACKEND. Use "
                "'sentence-transformers' or 'fastembed'."
            )

        return self._embedder

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """Generate normalized embeddings with the selected backend."""
        if not texts:
            return []

        embedder = self._get_embedder()

        if EMBEDDING_BACKEND == "fastembed":
            vectors = list(embedder.embed(texts))
        else:
            vectors = embedder.encode(
                texts,
                normalize_embeddings=True,
                batch_size=16,
                show_progress_bar=False,
            )

        normalized = []

        for vector in vectors:
            values = [float(value) for value in vector]
            norm = math.sqrt(sum(value * value for value in values))

            if norm == 0:
                normalized.append(values)
            else:
                normalized.append([value / norm for value in values])

        return normalized

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

        embeddings = self._encode(documents)

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

        total_count = self.collection.count()

        if total_count == 0:
            return []

        query_embedding = self._encode([query])[0]

        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, total_count),
            "include": ["documents", "metadatas", "distances"],
        }

        if document_id:
            query_args["where"] = {"document_id": document_id}

        results = self.collection.query(**query_args)

        result_documents = results.get("documents") or [[]]

        if not result_documents or not result_documents[0]:
            return []

        matches = []

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
