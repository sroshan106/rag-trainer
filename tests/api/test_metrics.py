import json

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.routes.metrics import frame_generator

client = TestClient(app)


def test_metrics_snapshot_has_cpu_memory_disk():
    resp = client.get("/api/metrics")

    assert resp.status_code == 200
    body = resp.json()
    assert "cpu" in body
    assert "memory" in body
    assert "disk" in body
    # gpu is optional -- absent when NVML can't be initialized on this host.


class _FakeRequest:
    """Disconnects after ``n`` polls, so the generator under test terminates.

    Driving ``EventSourceResponse`` itself needs a live ASGI connection to
    signal disconnect -- exercised for real via a running uvicorn server
    (see manual verification in the task report). This tests the generator's
    own logic in isolation: it stops polling once the client is gone.
    """

    def __init__(self, disconnect_after: int):
        self._n = disconnect_after
        self._polls = 0

    async def is_disconnected(self) -> bool:
        self._polls += 1
        return self._polls > self._n


@pytest.mark.anyio
async def test_frame_generator_yields_metrics_frames(monkeypatch):
    monkeypatch.setattr("src.api.routes.metrics.STREAM_INTERVAL_SECONDS", 0)
    request = _FakeRequest(disconnect_after=2)

    frames = [f async for f in frame_generator(request)]

    assert len(frames) == 2
    for frame in frames:
        assert frame["event"] == "metrics"
        payload = json.loads(frame["data"])
        assert "cpu" in payload


@pytest.mark.anyio
async def test_frame_generator_stops_immediately_if_already_disconnected():
    request = _FakeRequest(disconnect_after=0)

    frames = [f async for f in frame_generator(request)]

    assert frames == []


def test_health():
    resp = client.get("/api/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
