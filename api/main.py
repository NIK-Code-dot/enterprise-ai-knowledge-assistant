
from pathlib import Path
import hashlib
import re

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from src.document_reader import extract_pdf_text
from src.chunking import split_pages
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline

app = FastAPI(
    title="Enterprise AI Knowledge Assistant",
    description="Upload PDFs and ask questions using local AI and RAG.",
    version="1.0.0",
)

rag_pipeline = RAGPipeline()
store = rag_pipeline.store


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)
    document_id: str | None = None


class Source(BaseModel):
    source: str
    page: int
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "healthy", "service": "Enterprise AI Knowledge Assistant"}


@app.get("/documents")
def list_documents():
    return {"documents": store.list_documents()}


@app.post("/documents/upload")
async def upload_pdf(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if not file_bytes.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")

    document_id = store.make_document_id(filename, file_bytes)

    # Extract from a temporary file; don't overwrite other uploaded files.
    temp_dir = Path("artifacts") / "uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{hashlib.sha256(file_bytes).hexdigest()}.pdf"

    try:
        temp_path.write_bytes(file_bytes)
        pages = extract_pdf_text(str(temp_path))

        # Restore the original filename in source metadata.
        for page in pages:
            page["source"] = filename

        chunks = split_pages(pages)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No extractable text found. This PDF may require OCR.",
            )

        indexed = store.add_chunks(chunks, document_id=document_id)

        return {
            "message": "PDF uploaded and indexed successfully.",
            "document_id": document_id,
            "filename": filename,
            "pages_with_text": len(pages),
            "chunks_indexed": indexed,
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {error}",
        ) from error
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    try:
        result = rag_pipeline.answer(
            question=request.question,
            top_k=request.top_k,
            document_id=request.document_id,
        )
        return result
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to answer question: {error}",
        ) from error

