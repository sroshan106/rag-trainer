from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)

TEST_MODEL = "llama3.2:3b"


def _fake_ask_compare(q, model=None):
    return {
        "answer": f"grounded answer to {q}",
        "citations": [],
        "refused": False,
        "confidence": 0.83,
        "model": model,
        "latency_ms": 900.0,
        "rerank_ms": 40.0,
        "generate_ms": 800.0,
    }


def _fake_ask_direct(q, model=None):
    return {
        "answer": f"raw answer to {q}",
        "model": model,
        "latency_ms": 700.0,
        "eval_count": 50,
        "tokens_per_sec": 62.5,
    }


def test_compare_returns_both_sides(monkeypatch):
    monkeypatch.setattr("src.api.routes.benchmark.ask_compare", _fake_ask_compare)
    monkeypatch.setattr("src.api.routes.benchmark.ask_direct", _fake_ask_direct)

    resp = client.post("/api/benchmark/compare", json={"query": "what is this?", "model": TEST_MODEL})

    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == TEST_MODEL
    assert body["grounded"]["answer"] == "grounded answer to what is this?"
    assert body["grounded"]["rerank_ms"] == 40.0
    assert body["direct"]["answer"] == "raw answer to what is this?"
    assert body["direct"]["tokens_per_sec"] == 62.5


def test_compare_rejects_an_unknown_model():
    resp = client.post("/api/benchmark/compare", json={"query": "q", "model": "gpt-4"})

    assert resp.status_code == 422


def test_compare_rejects_when_no_model_given():
    resp = client.post("/api/benchmark/compare", json={"query": "q"})

    assert resp.status_code == 422


def test_compare_failure_surfaces_as_502(monkeypatch):
    def _boom(q, model=None):
        raise RuntimeError("ollama unreachable")

    monkeypatch.setattr("src.api.routes.benchmark.ask_compare", _boom)

    resp = client.post("/api/benchmark/compare", json={"query": "q", "model": TEST_MODEL})

    assert resp.status_code == 502
    assert "ollama unreachable" in resp.json()["detail"]
