"""The Ask view's backing route: one query in, one grounded answer out.

Delegates entirely to ``src.rag.graph.ask`` -- no retrieval or generation
logic lives here. Defined as a sync function deliberately: FastAPI runs sync
path operations in a worker thread, which is what keeps ``ask``'s blocking
Postgres/Ollama calls off the event loop without this route needing to know
anything about threading.
"""

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from src.api.schemas import CollectionStatus, QueryRequest, QueryResponse
from src.rag.graph import ask, ask_stream
from src.rag.model_catalog import list_installed
from src.rag.nodes import MODEL
from src.vectorstore.store import count_chunks

router = APIRouter(prefix="/api/query", tags=["query"])


@router.get("/models")
def list_models() -> dict:
    # Only models actually pulled -- a catalog entry mid-download or never
    # fetched must not be selectable here, see src.rag.model_catalog.
    return {"models": list_installed(), "default": MODEL}


@router.post("", response_model=QueryResponse)
def run_query(body: QueryRequest) -> dict:
    _validated_model(body.model)
    try:
        return ask(body.query, model=body.model)
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as a 502
        raise HTTPException(status_code=502, detail=f"query failed: {exc}") from exc


@router.get("/collection", response_model=CollectionStatus)
def collection_status() -> dict:
    """Whether there is anything to ask about at all.

    Without this the Ask view cannot tell an empty collection from a refusal:
    both produce an answer with no sources, and telling a user their question
    was out of scope when nothing has been ingested is the wrong instruction.
    """
    chunks = count_chunks()
    return {"chunks": chunks, "empty": chunks == 0}


def _validated_model(model: str | None) -> str | None:
    installed = list_installed()
    if model is not None and model not in installed:
        raise HTTPException(
            status_code=422,
            detail=f"unknown model {model!r} -- choose from {installed}",
        )
    return model


async def event_generator(request: Request, body: QueryRequest):
    """Bridge ``ask_stream``'s blocking generator onto the event loop.

    ``ask_stream`` does blocking work (Postgres, the cross-encoder, Ollama), so
    each step is pulled in a worker thread -- the same reason the non-streaming
    route is defined as a sync function. Closing the generator on the way out
    is what turns a client disconnect into a cancelled query instead of an
    orphaned one that keeps generating.
    """
    events = ask_stream(body.query, model=body.model)
    sentinel = object()
    try:
        while True:
            if await request.is_disconnected():
                break
            event = await asyncio.to_thread(next, events, sentinel)
            if event is sentinel:
                break
            yield {"event": event["type"], "data": json.dumps(event)}
    finally:
        events.close()


@router.post("/stream")
async def stream_query(request: Request, body: QueryRequest) -> EventSourceResponse:
    _validated_model(body.model)
    return EventSourceResponse(event_generator(request, body))
