"""Pydantic request/response models for the API."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    model: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    id: str | None = None


class CollectionStatus(BaseModel):
    chunks: int
    empty: bool


class HistoryEntry(BaseModel):
    id: str
    created_at: str
    query: str
    answer: str | None
    sources: list[str]
    refused: bool | None
    latency_ms: float | None = None
    rerank_ms: float | None = None
    generate_ms: float | None = None
    model: str | None = None
    status: str | None = None


class IngestFileEntry(BaseModel):
    id: str
    created_at: str
    filename: str
    stored_path: str
    sha256: str
    size_bytes: int
    documents: int | None = None
    chunk_ids: list[str] | None = None


class BenchmarkRequest(BaseModel):
    workers: int = Field(default=4, ge=1, le=32)
    sample: int | None = Field(default=None, ge=1, le=1000)
    use_cache: bool = True
    chunk_size: int = Field(default=10, ge=1, le=200)
    model: str | None = None
    test_files: list[str] | None = None


class BenchmarkTestFileEntry(BaseModel):
    id: str
    name: str
    filename: str
    builtin: bool = False
    questions: int
    suite_type: str
    created_at: str | None = None
    size_bytes: int | None = None


class PullModelRequest(BaseModel):
    model: str


class JobResponse(BaseModel):
    id: str
    kind: str
    status: str
    progress: float
    message: str
    result: object | None = None
    error: str | None = None
    params: dict | None = None
    created_at: float
    updated_at: float


class IngestResponse(BaseModel):
    documents: int | None = None
    chunks: int | None = None
    stdout: str
