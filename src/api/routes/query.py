"""The Ask view's backing route: one query in, one grounded answer out.
Delegates entirely to ``src.rag.graph.ask``."""

import asyncio
import json
import threading

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from src.api.schemas import CollectionStatus, QueryRequest, QueryResponse
from src.rag.graph import ask, ask_stream
from src.rag.model_catalog import list_installed
from src.vectorstore.store import count_chunks

router = APIRouter(prefix="/api/query", tags=["query"])


@router.get("/models")
def list_models() -> dict:
    # Only models actually pulled -- a catalog entry mid-download or never
    # fetched must not be selectable here, see src.rag.model_catalog. No
    # "default" key: there is no fallback model, the UI must make the user
    # pick one of these (and download one first if the list is empty).
    return {"models": list_installed()}


@router.post("", response_model=QueryResponse)
def run_query(body: QueryRequest) -> dict:
    model = _validated_model(body.model)
    try:
        return ask(body.query, model=model)
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


def _validated_model(model: str | None) -> str:
    installed = list_installed()
    if model not in installed:
        if not installed:
            raise HTTPException(
                status_code=422,
                detail="no chat model downloaded -- download one in Settings first",
            )
        raise HTTPException(
            status_code=422,
            detail=f"unknown model {model!r} -- choose from {installed}",
        )
    return model


async def event_generator(request: Request, body: QueryRequest):
    """Bridge ``ask_stream``'s blocking generator onto the event loop.

    ``ask_stream`` does blocking work (Postgres, the cross-encoder, Ollama), so
    it is driven on a worker thread. A thread-safe queue delivers events onto the
    async event loop. Disconnection stops the worker and closes the generator,
    cancelling Ollama generation cleanly.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop_event = threading.Event()
    sentinel = object()

    def run_worker():
        events = None
        try:
            events = ask_stream(body.query, model=body.model)
            for event in events:
                if stop_event.is_set():
                    break
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "error", "detail": str(exc)},
            )
        finally:
            if events is not None:
                events.close()
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    worker_thread = threading.Thread(target=run_worker, daemon=True)
    worker_thread.start()

    try:
        while True:
            if await request.is_disconnected():
                stop_event.set()
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue

            if event is sentinel:
                break
            yield {"event": event["type"], "data": json.dumps(event)}
    finally:
        stop_event.set()


@router.post("/stream")
async def stream_query(request: Request, body: QueryRequest) -> EventSourceResponse:
    # ``_validated_model`` asks Ollama over blocking HTTP (5s timeout). This
    # path operation is async, so calling it inline would stall the event loop
    # -- and every other request with it -- on each stream request.
    await asyncio.to_thread(_validated_model, body.model)
    return EventSourceResponse(event_generator(request, body))
