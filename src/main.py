import os
import sys
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(__file__))

from chunker import chunk_document
from embedder import embed_and_upload
from rag import ask

app = FastAPI(
    title="RAG Pipeline API",
    description="Document ingestion and Q&A API powered by Azure AI Search + Groq",
    version="1.0.0"
)

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer:   str
    sources:  list[str]

class IngestResponse(BaseModel):
    filename:       str
    chunks_indexed: int
    status:         str


@app.get("/health")
def health():
    return {"status": "ok", "service": "RAG Pipeline API"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    allowed = {".pdf", ".docx", ".html", ".jpg", ".png"}
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {allowed}"
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks = chunk_document(file_path)
        embed_and_upload(chunks)
        return IngestResponse(
            filename=file.filename,
            chunks_indexed=len(chunks),
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    try:
        result = ask(request.question)
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))