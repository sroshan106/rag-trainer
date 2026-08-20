"""Pydantic request/response models for the API."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    model: str | None = None


class Citation(BaseModel):
    """Where one grounded chunk came from. See src/ingestion/units.py."""

    file_id: str | None = None
    filename: str | None = None
    unit_kind: str | None = None
    unit_index: int | None = None
    label: str | None = None
    url: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    refused: bool = False
    confidence: float | None = None
    id: str | None = None


class CollectionStatus(BaseModel):
    chunks: int
    empty: bool


class HistoryEntry(BaseModel):
    id: str
    created_at: str
    query: str
    answer: str | None
    # Pre-citation rows still carry bare URL strings here; see history.py.
    sources: list[str] = []
    citations: list[Citation] = []
    refused: bool | None
    confidence: float | None = None
    latency_ms: float | None = None
    rerank_ms: float | None = None
    generate_ms: float | None = None
    model: str | None = None
    status: str | None = None


class IngestFileEntry(BaseModel):
    id: str
    created_at: str
    filename: str
    # stored_path is deliberately absent: it is a server filesystem path, and
    # the browser addresses documents by id through /api/documents instead.
    sha256: str
    size_bytes: int
    documents: int | None = None
    chunk_ids: list[str] | None = None


class DocumentMeta(BaseModel):
    """What a stored document is, before any of its text is fetched."""

    id: str
    filename: str
    extension: str
    unit_kind: str
    units: int | None = None
    size_bytes: int
    created_at: str
    chunks: int = 0
    # CSV header, when the viewer can render the document as a table.
    columns: list[str] | None = None


class UnitEntry(BaseModel):
    """One addressable piece of a document, as ingest read it."""

    index: int
    kind: str
    text: str
    label: str
    url: str | None = None
    key: str | None = None


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    model: str | None = None


class DirectCompareSide(BaseModel):
    """The raw-model half of a comparison -- no retrieval, no context."""

    answer: str
    latency_ms: float
    eval_count: int | None = None
    tokens_per_sec: float | None = None


class GroundedCompareSide(BaseModel):
    """The RAG-grounded half of a comparison -- same shape ``ask`` returns,
    plus the timings ``ask_stream``'s done event carries."""

    answer: str
    citations: list[Citation] = []
    refused: bool = False
    confidence: float | None = None
    latency_ms: float
    rerank_ms: float | None = None
    generate_ms: float | None = None


class CompareResponse(BaseModel):
    """Same query, same model, retrieval on vs off -- side by side. Neither
    side is written to query history."""

    model: str
    grounded: GroundedCompareSide
    direct: DirectCompareSide


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
