"""The System view's backing route: one-shot metrics plus a 1Hz SSE stream.

SSE rather than WebSocket because the flow is strictly server-to-client
(ui_plan.md, "Transport") -- it reconnects automatically and needs no extra
protocol handling for a stream that never receives client messages.
"""

import asyncio
import json

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from src.observability.sysmetrics import collect_all

router = APIRouter(prefix="/api/metrics", tags=["metrics"])

STREAM_INTERVAL_SECONDS = 1.0


@router.get("")
def get_metrics() -> dict:
    return collect_all()


async def frame_generator(request: Request):
    """The stream body, factored out so tests can drive it without a live socket."""
    while True:
        if await request.is_disconnected():
            break
        # collect_all() does blocking syscalls (psutil, NVML); off the
        # event loop so one slow sample can't stall other requests.
        frame = await asyncio.to_thread(collect_all)
        yield {"event": "metrics", "data": json.dumps(frame)}
        await asyncio.sleep(STREAM_INTERVAL_SECONDS)


@router.get("/stream")
async def stream_metrics(request: Request) -> EventSourceResponse:
    return EventSourceResponse(frame_generator(request))
