from pydantic import BaseModel, Field
from typing import List, Optional, Dict


# ---------- Shared Types ----------

class Metadata(BaseModel):
    category: Optional[str] = Field(None, description="Category label for the document")
    tags: Optional[List[str]] = Field(None, description="Optional list of tags")


# ---------- /ingest ----------

class IngestRequest(BaseModel):
    source: str = Field(..., description="Data source type (e.g., 'local', 'upload')")
    file_path: Optional[str] = Field(
        None, description="Path to the file on disk if source='local'"
    )
    metadata: Optional[Metadata] = None


class IngestResponse(BaseModel):
    status: str = Field(..., example="success")
    document_id: str = Field(..., description="Unique identifier of the ingested document")
    chunks_stored: int = Field(..., description="Number of text chunks created from the document")


# ---------- /list_documents ----------

class DocumentInfo(BaseModel):
    id: str
    name: str
    chunks: int
    metadata: Optional[Metadata]


class ListDocumentsResponse(BaseModel):
    documents: List[DocumentInfo]


# ---------- /query ----------

class QueryRequest(BaseModel):
    query: str = Field(..., description="The search query or user question")
    top_k: int = Field(3, description="Number of top results to return")


class QueryResult(BaseModel):
    document_id: str
    score: float
    text: str


class QueryResponse(BaseModel):
    results: List[QueryResult]


# ---------- /generate ----------

class LLMConfig(BaseModel):
    provider: str = Field(..., example="ollama", description="LLM provider name")
    model: str = Field(..., example="mistral", description="Model identifier")
    temperature: Optional[float] = Field(0.2, description="Sampling temperature")
    api_key: Optional[str] = Field(None, description="Optional API key if needed")


class GenerateRequest(BaseModel):
    query: str
    top_k: int = Field(3, description="Number of retrieval results to use")
    llm: LLMConfig


class ContextUsed(BaseModel):
    document_id: str
    score: float


class GenerateResponse(BaseModel):
    answer: str
    context_used: List[ContextUsed]


# ---------- /health ----------

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: int
