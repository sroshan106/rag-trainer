"""Pydantic request/response models for the API.

Kept separate from the routes so a route file stays about wiring, not shape.
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    # Omitted means the default model; the route validates it against
    # AVAILABLE_MODELS rather than trusting whatever string arrives.
    model: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    # None when history recording is disabled or the write failed; the answer
    # is still returned either way.
    id: str | None = None


class HistoryEntry(BaseModel):
    id: str
    created_at: str
    query: str
    answer: str
    sources: list[str]
    refused: bool
    latency_ms: float | None = None
    model: str | None = None


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
    # Mirrors tests.benchmark.run_benchmark.run_all's signature -- see
    # DEFAULT_WORKERS there for why 4 is the default on this GPU.
    workers: int = Field(default=4, ge=1, le=32)
    sample: int | None = Field(default=None, ge=1, le=1000)
    use_cache: bool = True


class JobResponse(BaseModel):
    id: str
    kind: str
    status: str
    progress: float
    message: str
    result: object | None = None
    error: str | None = None
    created_at: float
    updated_at: float


class IngestResponse(BaseModel):
    documents: int | None = None
    chunks: int | None = None
    stdout: str
