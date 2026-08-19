"""Pydantic request/response models for the API.

Kept separate from the routes so a route file stays about wiring, not shape.
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


class BenchmarkRequest(BaseModel):
    # Mirrors tests.benchmark.run_benchmark.run_all's signature -- see
    # DEFAULT_WORKERS there for why 8 is the default on this GPU.
    workers: int = Field(default=8, ge=1, le=32)
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
