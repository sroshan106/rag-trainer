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


class _FakeRequest:

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


def test_get_logs(monkeypatch):
    from src.observability import logging as obs_logging
    obs_logging.log("info", "test log entry", detail="unit test")

    resp = client.get("/api/metrics/logs?limit=5")
    assert resp.status_code == 200
    entries = resp.json()
    assert isinstance(entries, list)
    assert any(e.get("message") == "test log entry" for e in entries)
