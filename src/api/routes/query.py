"""The Ask view's backing route: one query in, one grounded answer out.

Delegates entirely to ``src.rag.graph.ask`` -- no retrieval or generation
logic lives here. Defined as a sync function deliberately: FastAPI runs sync
path operations in a worker thread, which is what keeps ``ask``'s blocking
Postgres/Ollama calls off the event loop without this route needing to know
anything about threading.
"""

from fastapi import APIRouter, HTTPException

from src.api.schemas import QueryRequest, QueryResponse
from src.rag.graph import ask

router = APIRouter(prefix="/api/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def run_query(body: QueryRequest) -> dict:
    try:
        return ask(body.query)
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as a 502
        raise HTTPException(status_code=502, detail=f"query failed: {exc}") from exc
