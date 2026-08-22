import asyncio
import json

from fastapi import APIRouter, Query, Request
from sse_starlette.sse import EventSourceResponse

from src.observability.logging import tail
from src.observability.sysmetrics import collect_all

router = APIRouter(prefix="/api/metrics", tags=["metrics"])

STREAM_INTERVAL_SECONDS = 1.0


@router.get("")
def get_metrics() -> dict:
    return collect_all()


@router.get("/logs")
def get_logs(
    limit: int = Query(default=100, ge=1, le=1000),
    level: str | None = None,
    q: str | None = None,
) -> list[dict]:
    return tail(limit=limit, level=level, query=q)


async def frame_generator(request: Request):
    while True:
        if await request.is_disconnected():
            break
        frame = await asyncio.to_thread(collect_all)
        yield {"event": "metrics", "data": json.dumps(frame)}
        await asyncio.sleep(STREAM_INTERVAL_SECONDS)


@router.get("/stream")
async def stream_metrics(request: Request) -> EventSourceResponse:
    return EventSourceResponse(frame_generator(request))
