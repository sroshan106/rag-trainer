from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_list_models_returns_expected_structure(monkeypatch):
    monkeypatch.setattr("src.rag.model_catalog._installed_names", lambda: {"llama3.2:3b", "nomic-embed-text"})
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "catalog" in data
    assert "installed" in data
    assert "llama3.2:3b" in data["installed"]


def test_delete_ollama_model(monkeypatch):
    deleted = []

    def fake_delete_model(model: str):
        deleted.append(model)

    monkeypatch.setattr("src.rag.model_catalog.delete_model", fake_delete_model)

    resp = client.delete("/api/models/llama3.2:3b")
    assert resp.status_code == 200
    assert resp.json() == {"status": "deleted", "model": "llama3.2:3b"}
    assert deleted == ["llama3.2:3b"]


def test_delete_reranker_model_with_slashes(monkeypatch):
    deleted = []

    def fake_delete_model(model: str):
        deleted.append(model)

    monkeypatch.setattr("src.rag.model_catalog.delete_model", fake_delete_model)

    resp = client.delete("/api/models/cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert resp.status_code == 200
    assert resp.json() == {"status": "deleted", "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
    assert deleted == ["cross-encoder/ms-marco-MiniLM-L-6-v2"]
