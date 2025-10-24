from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import time

from rag_module.core.ingest import process_file
from rag_module.api.schemas import (
    IngestRequest,
    IngestResponse,
    ListDocumentsResponse,
    DocumentInfo,
    QueryRequest,
    QueryResponse,
    QueryResult,
    GenerateRequest,
    GenerateResponse,
    ContextUsed,
    HealthResponse,
)

from rag_module.core.retriever import retrieve_top_k
from rag_module.core.generator import generate_answer as core_generate_answer


# from rag_module.core.store import _VECTOR_STORE

router = APIRouter()

# Temporary in-memory mock storage
DOCUMENTS_DB = []
START_TIME = time.time()


# ---------- /ingest ----------
@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(req: IngestRequest):
    """
    Register a document for retrieval.
    For now, simulate ingestion and return mock data.
    """
    if not req.file_path:
        raise HTTPException(status_code=400, detail="file_path is required")
    
    try:
        doc_id = process_file(req.file_path, req.metadata.dict() if req.metadata else {})
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=f"File not found: {e}")
    except ImportError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return IngestResponse(status="success", document_id=doc_id, chunks_stored=42)


# ---------- /list_documents ----------
@router.get("/list_documents", response_model=ListDocumentsResponse)
async def list_documents():
    """Return all currently known documents."""
    docs = [
        DocumentInfo(
            id=d["id"],
            name=d["name"],
            chunks=d["chunks"],
            metadata=d["metadata"]
        )
        for d in DOCUMENTS_DB
    ]
    return ListDocumentsResponse(documents=docs)


# ---------- /query ----------
@router.post("/query", response_model=QueryResponse)
async def query_docs(req: QueryRequest):
    """
    Retrieve top-k context snippets for a user query.
    Currently returns placeholder results.
    """
    retrieved = retrieve_top_k(req.query, req.top_k)
    return QueryResponse(
        results=[
            QueryResult(document_id=r["doc_id"], score=r["score"], text=r["text"])
            for r in retrieved
        ]
    )


# ---------- /generate ----------
@router.post("/generate", response_model=GenerateResponse)
async def generate_answer(req: GenerateRequest):
    """
    Combine retrieval + LLM to answer a question.
    Currently uses placeholders.
    """
    retrieved = retrieve_top_k(req.query, req.top_k)
    answer = core_generate_answer(req.query, retrieved, req.llm.model)
    context_used = [
        ContextUsed(document_id=r["doc_id"], score=r["score"]) for r in retrieved
    ]
    return GenerateResponse(answer=answer, context_used=context_used)


# ---------- /health ----------
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Simple uptime check for service health."""
    uptime = int(time.time() - START_TIME)
    return HealthResponse(status="ok", uptime_seconds=uptime)

@router.get("/debug/store_size")
def store_size():
    col = _client.get_or_create_collection("rag_docs")
    return {"chunks": col.count()}