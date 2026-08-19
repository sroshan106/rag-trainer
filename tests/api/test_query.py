from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_query_returns_answer_and_sources(monkeypatch):
    monkeypatch.setattr(
        "src.api.routes.query.ask",
        lambda q, model=None: {"answer": f"answer to {q}", "sources": ["http://example.com"]},
    )

    resp = client.post("/api/query", json={"query": "what is this?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "answer to what is this?"
    assert body["sources"] == ["http://example.com"]


def test_query_passes_the_chosen_model_through(monkeypatch):
    seen = {}

    def fake_ask(q, model=None):
        seen["model"] = model
        return {"answer": "a", "sources": []}

    monkeypatch.setattr("src.api.routes.query.ask", fake_ask)

    resp = client.post("/api/query", json={"query": "q", "model": "qwen3:4b"})

    assert resp.status_code == 200
    assert seen["model"] == "qwen3:4b"


def test_query_rejects_an_unknown_model():
    resp = client.post("/api/query", json={"query": "q", "model": "gpt-4"})

    assert resp.status_code == 422


def test_list_models_reports_available_and_default():
    resp = client.get("/api/query/models")

    assert resp.status_code == 200
    body = resp.json()
    assert "llama3.2:3b" in body["models"]
    assert body["default"] == "llama3.2:3b"


def test_query_rejects_empty_string():
    resp = client.post("/api/query", json={"query": ""})

    assert resp.status_code == 422


def test_query_failure_surfaces_as_502(monkeypatch):
    def _boom(q, model=None):
        raise RuntimeError("ollama unreachable")

    monkeypatch.setattr("src.api.routes.query.ask", _boom)

    resp = client.post("/api/query", json={"query": "hello"})

    assert resp.status_code == 502
    assert "ollama unreachable" in resp.json()["detail"]
