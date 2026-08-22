import asyncio
import json
import threading

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from src.api.deps import validated_model
from src.api.schemas import CollectionStatus, QueryRequest, QueryResponse
from src.rag.graph import ask, ask_stream
from src.rag.model_catalog import list_installed
from src.vectorstore.store import count_chunks

router = APIRouter(prefix="/api/query", tags=["query"])


@router.get("/models")
def list_models() -> dict:
    return {"models": list_installed()}


@router.post("", response_model=QueryResponse)
def run_query(body: QueryRequest) -> dict:
    model = validated_model(body.model)
    try:
        return ask(body.query, model=model)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"query failed: {exc}") from exc


@router.get("/collection", response_model=CollectionStatus)
def collection_status() -> dict:
    chunks = count_chunks()
    return {"chunks": chunks, "empty": chunks == 0}


async def event_generator(request: Request, body: QueryRequest):
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
        except Exception as exc:
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
    await asyncio.to_thread(validated_model, body.model)
    return EventSourceResponse(event_generator(request, body))
