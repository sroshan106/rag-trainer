import json

from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)

# No default model exists anymore -- every request below must name one
# explicitly. Matches what src.rag.model_catalog.list_installed() actually
# reports against the live Ollama instance these tests run against.
TEST_MODEL = "llama3.2:3b"


CITATION = {
    "file_id": "file-1",
    "filename": "corpus.csv",
    "unit_kind": "row",
    "unit_index": 42,
    "label": "row 42",
    "url": None,
}


def test_query_returns_answer_and_citations(monkeypatch):
    monkeypatch.setattr(
        "src.api.routes.query.ask",
        lambda q, model=None: {
            "answer": f"answer to {q}",
            "citations": [CITATION],
            "refused": False,
            "confidence": 0.83,
        },
    )

    resp = client.post("/api/query", json={"query": "what is this?", "model": TEST_MODEL})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "answer to what is this?"
    assert body["citations"] == [CITATION]
    assert body["refused"] is False
    assert body["confidence"] == 0.83


def test_query_passes_the_chosen_model_through(monkeypatch):
    seen = {}

    def fake_ask(q, model=None):
        seen["model"] = model
        return {"answer": "a", "citations": [], "refused": False, "confidence": 0.0}

    monkeypatch.setattr("src.api.routes.query.ask", fake_ask)
    monkeypatch.setattr("src.api.routes.query.list_installed", lambda: ["qwen3:4b"])

    resp = client.post("/api/query", json={"query": "q", "model": "qwen3:4b"})

    assert resp.status_code == 200
    assert seen["model"] == "qwen3:4b"


def test_query_rejects_an_unknown_model():
    resp = client.post("/api/query", json={"query": "q", "model": "gpt-4"})

    assert resp.status_code == 422


def test_query_rejects_when_no_model_given():
    # No fallback -- a query without a model is a 422, not a silent default.
    resp = client.post("/api/query", json={"query": "q"})

    assert resp.status_code == 422


def test_list_models_reports_only_installed_models():
    resp = client.get("/api/query/models")

    assert resp.status_code == 200
    body = resp.json()
    assert "llama3.2:3b" in body["models"]
    assert "default" not in body


def test_models_catalog_endpoint_includes_requirements():
    resp = client.get("/api/models")

    assert resp.status_code == 200
    body = resp.json()
    assert "catalog" in body
    assert "model_info" in body
    assert "llama3.2:3b" in body["model_info"]
    info = body["model_info"]["llama3.2:3b"]
    assert "min_vram" in info
    assert "disk_size" in info
    assert "params" in info


def test_query_rejects_empty_string():
    resp = client.post("/api/query", json={"query": "", "model": TEST_MODEL})

    assert resp.status_code == 422


def test_query_failure_surfaces_as_502(monkeypatch):
    def _boom(q, model=None):
        raise RuntimeError("ollama unreachable")

    monkeypatch.setattr("src.api.routes.query.ask", _boom)

    resp = client.post("/api/query", json={"query": "hello", "model": TEST_MODEL})

    assert resp.status_code == 502
    assert "ollama unreachable" in resp.json()["detail"]


def _sse_events(text):
    """Pull the JSON payloads out of an SSE body.

    The whole body arrives at once here -- TestClient runs the endpoint to
    completion -- so the events can be parsed after the fact rather than
    incrementally.
    """
    return [
        json.loads(line[len("data:"):].strip())
        for line in text.splitlines()
        if line.startswith("data:")
    ]


def test_stream_emits_the_events_ask_stream_produced(monkeypatch):
    def fake_ask_stream(query, model=None):
        yield {"type": "stage", "stage": "retrieve"}
        yield {"type": "token", "text": f"answer to {query}"}
        yield {
            "type": "done",
            "id": "e1",
            "answer": "done",
            "citations": [CITATION],
            "refused": False,
            "confidence": 0.83,
        }

    monkeypatch.setattr("src.api.routes.query.ask_stream", fake_ask_stream)

    resp = client.post("/api/query/stream", json={"query": "what is this?", "model": TEST_MODEL})

    assert resp.status_code == 200
    events = _sse_events(resp.text)
    assert [e["type"] for e in events] == ["stage", "token", "done"]
    assert events[1]["text"] == "answer to what is this?"
    assert events[2]["citations"] == [CITATION]
    assert events[2]["confidence"] == 0.83


def test_stream_names_each_sse_event_after_its_type(monkeypatch):
    def fake_ask_stream(query, model=None):
        yield {"type": "done", "answer": "a", "citations": [], "refused": False}

    monkeypatch.setattr("src.api.routes.query.ask_stream", fake_ask_stream)

    resp = client.post("/api/query/stream", json={"query": "q", "model": TEST_MODEL})

    assert "event: done" in resp.text


def test_stream_passes_the_chosen_model_through(monkeypatch):
    seen = {}

    def fake_ask_stream(query, model=None):
        seen["model"] = model
        yield {"type": "done", "answer": "a", "citations": [], "refused": False}

    monkeypatch.setattr("src.api.routes.query.ask_stream", fake_ask_stream)
    monkeypatch.setattr("src.api.routes.query.list_installed", lambda: ["qwen3:4b"])

    client.post("/api/query/stream", json={"query": "q", "model": "qwen3:4b"})

    assert seen["model"] == "qwen3:4b"


def test_stream_rejects_an_unknown_model():
    resp = client.post("/api/query/stream", json={"query": "q", "model": "gpt-4"})

    assert resp.status_code == 422


def test_stream_rejects_when_no_model_given():
    resp = client.post("/api/query/stream", json={"query": "q"})

    assert resp.status_code == 422


def test_stream_forwards_an_error_event(monkeypatch):
    def fake_ask_stream(query, model=None):
        yield {"type": "error", "detail": "ollama unreachable"}

    monkeypatch.setattr("src.api.routes.query.ask_stream", fake_ask_stream)

    events = _sse_events(
        client.post("/api/query/stream", json={"query": "q", "model": TEST_MODEL}).text
    )

    assert events == [{"type": "error", "detail": "ollama unreachable"}]


def test_collection_reports_the_chunk_count(monkeypatch):
    monkeypatch.setattr("src.api.routes.query.count_chunks", lambda: 42)

    resp = client.get("/api/query/collection")

    assert resp.status_code == 200
    assert resp.json() == {"chunks": 42, "empty": False}


def test_collection_is_empty_when_nothing_is_ingested(monkeypatch):
    monkeypatch.setattr("src.api.routes.query.count_chunks", lambda: 0)

    assert client.get("/api/query/collection").json() == {"chunks": 0, "empty": True}
