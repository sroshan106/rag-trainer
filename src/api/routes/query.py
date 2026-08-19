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
from src.rag.nodes import AVAILABLE_MODELS, MODEL

router = APIRouter(prefix="/api/query", tags=["query"])


@router.get("/models")
def list_models() -> dict:
    return {"models": list(AVAILABLE_MODELS), "default": MODEL}


@router.post("", response_model=QueryResponse)
def run_query(body: QueryRequest) -> dict:
    if body.model is not None and body.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=422,
            detail=f"unknown model {body.model!r} -- choose from {list(AVAILABLE_MODELS)}",
        )
    try:
        return ask(body.query, model=body.model)
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as a 502
        raise HTTPException(status_code=502, detail=f"query failed: {exc}") from exc
