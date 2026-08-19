from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_query_returns_answer_and_sources(monkeypatch):
    monkeypatch.setattr(
        "src.api.routes.query.ask",
        lambda q: {"answer": f"answer to {q}", "sources": ["http://example.com"]},
    )

    resp = client.post("/api/query", json={"query": "what is this?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "answer to what is this?"
    assert body["sources"] == ["http://example.com"]


def test_query_rejects_empty_string():
    resp = client.post("/api/query", json={"query": ""})

    assert resp.status_code == 422


def test_query_failure_surfaces_as_502(monkeypatch):
    def _boom(q):
        raise RuntimeError("ollama unreachable")

    monkeypatch.setattr("src.api.routes.query.ask", _boom)

    resp = client.post("/api/query", json={"query": "hello"})

    assert resp.status_code == 502
    assert "ollama unreachable" in resp.json()["detail"]
