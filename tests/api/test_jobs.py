import threading
import time
import uuid

from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def _wait_until(job_id: str, predicate, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if predicate(body):
            return body
        time.sleep(0.02)
    raise AssertionError("job did not reach the expected state in time")


def _wait_for_job(job_id: str, timeout: float = 5.0) -> dict:
    return _wait_until(
        job_id, lambda b: b["status"] in ("done", "failed", "cancelled"), timeout
    )


def test_ingest_runs_as_background_job_and_reports_counts(monkeypatch, tmp_path):
    def fake_ingest(path, progress=None, splitter=None):
        if progress:
            progress(0.5, "embedding")
        return {
            "path": str(path),
            "documents": 3,
            "chunks": 9,
            "chunk_ids": ["c1"],
            "splitter": splitter,
            "index_built": False,
        }

    monkeypatch.setattr("src.api.routes.ingest.ingest", fake_ingest)
    monkeypatch.setattr("src.api.routes.ingest.UPLOAD_DIR", tmp_path / "uploads")

    # Content is unique per run -- this test hits the real dedup table (no
    # file_history mocking here), and a repeated hash would 409 instead of
    # exercising the job path.
    csv = f"index,source_url,text\n1,http://example.com,hello {uuid.uuid4().hex}\n"
    resp = client.post("/api/ingest/upload", files={"file": ("mine.csv", csv, "text/csv")})
    assert resp.status_code == 202
    job_id = resp.json()["id"]

    final = _wait_for_job(job_id)
    assert final["status"] == "done"
    assert final["result"]["documents"] == 3
    assert final["result"]["chunks"] == 9


def test_benchmark_runs_as_background_job(monkeypatch):
    fake_results = [{"name": "single_passage", "n": 10, "recall@5": 0.9}]
    monkeypatch.setattr(
        "src.api.routes.benchmark.run_all", lambda *a, **kw: fake_results
    )

    resp = client.post("/api/benchmark", json={"workers": 2, "sample": 5, "model": "llama3.2:3b"})
    assert resp.status_code == 202
    job_id = resp.json()["id"]

    final = _wait_for_job(job_id)
    assert final["status"] == "done"
    assert final["result"] == fake_results


def test_benchmark_publishes_partial_results_while_running(monkeypatch):
    """A run that reports progress must expose metrics before it finishes --
    otherwise every config change costs a full run before it can be judged."""
    partial = [{"name": "single_passage", "n": 10, "recall@5": 0.5}]
    seen = threading.Event()

    def fake_run_all(workers, sample, use_cache, model, chunk_size, on_progress, should_stop):
        on_progress(partial, 10, 30)
        seen.wait(timeout=5)
        return [{"name": "single_passage", "n": 30, "recall@5": 0.7}]

    monkeypatch.setattr("src.api.routes.benchmark.run_all", fake_run_all)

    job_id = client.post("/api/benchmark", json={"model": "llama3.2:3b"}).json()["id"]
    body = _wait_until(job_id, lambda b: b["result"] is not None)
    assert body["status"] == "running"
    assert body["result"] == partial
    assert body["progress"] == 10 / 30

    seen.set()
    assert _wait_for_job(job_id)["result"][0]["n"] == 30


def test_benchmark_cancel_keeps_partial_result(monkeypatch):
    partial = [{"name": "single_passage", "n": 10, "recall@5": 0.5}]

    def fake_run_all(workers, sample, use_cache, model, chunk_size, on_progress, should_stop):
        on_progress(partial, 10, 30)
        while not should_stop():
            time.sleep(0.01)
        # Mirrors the real runner: stop early, return what was scored so far.
        return partial

    monkeypatch.setattr("src.api.routes.benchmark.run_all", fake_run_all)

    job_id = client.post("/api/benchmark", json={"model": "llama3.2:3b"}).json()["id"]
    _wait_until(job_id, lambda b: b["result"] is not None)

    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 200

    final = _wait_for_job(job_id)
    assert final["status"] == "cancelled"
    assert final["result"] == partial


def test_cancel_unknown_job_returns_404():
    assert client.post("/api/jobs/does-not-exist/cancel").status_code == 404


def test_cancel_finished_job_returns_409(monkeypatch):
    monkeypatch.setattr("src.api.routes.benchmark.run_all", lambda *a, **kw: [])
    job_id = client.post("/api/benchmark", json={"model": "llama3.2:3b"}).json()["id"]
    _wait_for_job(job_id)

    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 409


def test_benchmark_rejects_unknown_model(monkeypatch):
    monkeypatch.setattr("src.api.routes.benchmark.run_all", lambda *a, **kw: [])

    resp = client.post("/api/benchmark", json={"model": "gpt-nope"})

    assert resp.status_code == 422


def test_benchmark_unavailable_returns_503(monkeypatch):
    monkeypatch.setattr("src.api.routes.benchmark.run_all", None)

    resp = client.post("/api/benchmark", json={})

    assert resp.status_code == 503


def test_job_not_found_returns_404():
    resp = client.get("/api/jobs/does-not-exist")

    assert resp.status_code == 404
