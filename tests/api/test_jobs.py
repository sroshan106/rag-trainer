import time

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


def test_ingest_runs_as_background_job_and_reports_counts(monkeypatch):
    def fake_pipeline():
        print("loaded 3 documents")
        print("split into 9 chunks")
        print("ingest complete")

    monkeypatch.setattr("src.api.routes.ingest.run_pipeline", fake_pipeline)

    resp = client.post("/api/ingest")
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
