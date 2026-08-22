import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def mock_installed_models(monkeypatch):
    installed = ["llama3.2:3b", "qwen2.5:3b", "gemma2:2b"]
    monkeypatch.setattr("src.rag.model_catalog.list_installed", lambda: list(installed))
    monkeypatch.setattr("src.api.routes.query.list_installed", lambda: list(installed))
    monkeypatch.setattr("src.api.routes.benchmark.list_installed", lambda: list(installed))
