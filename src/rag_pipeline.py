
import os
import sys
import traceback

import requests

from src.vector_store import VectorStore


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
).strip()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/chat",
).strip()

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3.5:4b",
).strip()


class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStore()

    def answer(
        self,
        question: str,
        document_id: str | None = None,
    ) -> dict:
        question = question.strip()

        if not question:
            return {
                "answer": "Please enter a question.",
                "sources": [],
            }

        # Retrieve relevant document chunks.
        try:
            matches = self.vector_store.search(
                question,
                top_k=4,
                document_id=document_id,
            )
        except Exception as error:
            print(
                f"Document retrieval failed: {error}",
                file=sys.stderr,
                flush=True,
            )
            raise RuntimeError(
                f"Document retrieval failed: {error}"
            ) from error

        if not matches:
            return {
                "answer": (
                    "I could not find relevant information in the "
                    "uploaded documents. Please upload a document "
                    "containing the information and try again."
                ),
                "sources": [],
            }

        # Build context and source references.
        context_parts = []
        sources = []

        for match in matches:
            source_name = str(match.get("source", "unknown"))
            page = match.get("page", 0)
            chunk_text = str(match.get("text", "")).strip()

            if not chunk_text:
                continue

            context_parts.append(
                f"[Source: {source_name}, Page: {page}]\n"
                f"{chunk_text}"
            )

            source_item = {
                "source": source_name,
                "page": page,
            }

            if source_item not in sources:
                sources.append(source_item)

        if not context_parts:
            return {
                "answer": (
                    "No readable content was found in the retrieved "
                    "document sections."
                ),
                "sources": [],
            }

        context = "\n\n".join(context_parts)

        system_prompt = (
            "You are an Enterprise AI Knowledge Assistant. "
            "Answer the user's question using only the supplied "
            "document context. Do not invent facts. If the context "
            "does not contain the answer, clearly say that the "
            "uploaded documents do not provide enough information. "
            "When possible, mention the source filename and page "
            "number supporting your answer."
        )

        user_prompt = (
            f"Document context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer clearly and accurately using the context above."
        )

        # Use Groq in cloud deployment when an API key is configured.
        # Otherwise, fall back to the local Ollama service.
        try:
            if GROQ_API_KEY:
                response = requests.post(
                    GROQ_API_URL,
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": GROQ_MODEL,
                        "messages": [
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {
                                "role": "user",
                                "content": user_prompt,
                            },
                        ],
                        "temperature": 0.2,
                    },
                    timeout=90,
                )

                response.raise_for_status()
                response_data = response.json()

                answer_text = (
                    response_data["choices"][0]["message"]["content"]
                )

            else:
                response = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": [
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {
                                "role": "user",
                                "content": user_prompt,
                            },
                        ],
                        "stream": False,
                    },
                    timeout=120,
                )

                response.raise_for_status()
                response_data = response.json()
                answer_text = response_data["message"]["content"]

            if not isinstance(answer_text, str) or not answer_text.strip():
                raise ValueError(
                    "The AI provider returned an empty answer."
                )

        except Exception as error:
            # Print the real exception in Render backend logs.
            print(
                f"RAG answer generation failed "
                f"(provider={'Groq' if GROQ_API_KEY else 'Ollama'}): "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )

            if isinstance(error, requests.HTTPError):
                # Log the provider's response body without logging
                # request headers or the API key.
                try:
                    print(
                        "Provider response: "
                        f"{error.response.text[:2000]}",
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception:
                    pass

            traceback.print_exc(file=sys.stderr)

            provider = "Groq" if GROQ_API_KEY else "Ollama"

            raise RuntimeError(
                f"Could not generate an answer with {provider}: "
                f"{type(error).__name__}: {error}"
            ) from error

        return {
            "answer": answer_text.strip(),
            "sources": sources,
        }


if __name__ == "__main__":
    pipeline = RAGPipeline()

    print("Enterprise AI Knowledge Assistant")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            question = input("Question: ").strip()

            if question.lower() in {"exit", "quit"}:
                break

            if not question:
                continue

            result = pipeline.answer(question)

            print("\nAnswer:")
            print(result["answer"])

            print("\nSources:")
            for source in result["sources"]:
                print(
                    f"- {source['source']} "
                    f"(Page {source['page']})"
                )

            print()

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        except Exception as error:
            print(f"\nError: {error}\n")
