from langchain_core.documents import Document

from src.vectorstore import store


class FakePGVector:
    last_from_documents_kwargs = None
    last_init_kwargs = None

    def __init__(self, **kwargs):
        FakePGVector.last_init_kwargs = kwargs

    @classmethod
    def from_documents(cls, **kwargs):
        cls.last_from_documents_kwargs = kwargs
        return cls()


def test_build_vectorstore_passes_collection_and_connection(monkeypatch):
    monkeypatch.setattr(store, "PGVector", FakePGVector)
    chunks = [Document(page_content="x", metadata={})]

    store.build_vectorstore(chunks, connection="postgresql://test")

    kwargs = FakePGVector.last_from_documents_kwargs
    assert kwargs["documents"] == chunks
    assert kwargs["collection_name"] == store.COLLECTION_NAME
    assert kwargs["connection"] == "postgresql://test"


def test_build_vectorstore_falls_back_to_env(monkeypatch):
    monkeypatch.setattr(store, "PGVector", FakePGVector)
    monkeypatch.setenv("DATABASE_URL", "postgresql://from-env")

    store.build_vectorstore([], connection=None)

    assert FakePGVector.last_from_documents_kwargs["connection"] == "postgresql://from-env"


def test_load_vectorstore_uses_collection_name(monkeypatch):
    monkeypatch.setattr(store, "PGVector", FakePGVector)

    store.load_vectorstore(connection="postgresql://test")

    kwargs = FakePGVector.last_init_kwargs
    assert kwargs["collection_name"] == store.COLLECTION_NAME
    assert kwargs["connection"] == "postgresql://test"
