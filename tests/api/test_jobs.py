import time
import uuid

from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def _wait_for_job(job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/jobs/{job_id}")
        body = resp.json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError("job did not finish in time")


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
    monkeypatch.setattr("src.api.routes.benchmark.run_all", lambda w, s, use_cache: fake_results)

    resp = client.post("/api/benchmark", json={"workers": 2, "sample": 5})
    assert resp.status_code == 202
    job_id = resp.json()["id"]

    final = _wait_for_job(job_id)
    assert final["status"] == "done"
    assert final["result"] == fake_results


def test_benchmark_unavailable_returns_503(monkeypatch):
    monkeypatch.setattr("src.api.routes.benchmark.run_all", None)

    resp = client.post("/api/benchmark", json={})

    assert resp.status_code == 503


def test_job_not_found_returns_404():
    resp = client.get("/api/jobs/does-not-exist")

    assert resp.status_code == 404
