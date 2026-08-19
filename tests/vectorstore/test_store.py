from langchain_core.documents import Document

from src.vectorstore import store


class FakePGVector:
    last_from_documents_kwargs = None
    last_init_kwargs = None
    last_delete_ids = None

    def __init__(self, **kwargs):
        FakePGVector.last_init_kwargs = kwargs

    @classmethod
    def from_documents(cls, **kwargs):
        cls.last_from_documents_kwargs = kwargs
        return cls()

    def delete(self, ids=None, **kwargs):
        FakePGVector.last_delete_ids = ids


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


def test_build_vectorstore_generates_and_returns_ids(monkeypatch):
    monkeypatch.setattr(store, "PGVector", FakePGVector)
    chunks = [Document(page_content="x", metadata={}), Document(page_content="y", metadata={})]

    ids = store.build_vectorstore(chunks, connection="postgresql://test")

    assert len(ids) == 2
    assert len(set(ids)) == 2
    assert FakePGVector.last_from_documents_kwargs["ids"] == ids


def test_build_vectorstore_uses_provided_ids(monkeypatch):
    monkeypatch.setattr(store, "PGVector", FakePGVector)
    chunks = [Document(page_content="x", metadata={})]

    ids = store.build_vectorstore(chunks, connection="postgresql://test", ids=["fixed-id"])

    assert ids == ["fixed-id"]
    assert FakePGVector.last_from_documents_kwargs["ids"] == ["fixed-id"]


def test_delete_chunks_deletes_by_id(monkeypatch):
    monkeypatch.setattr(store, "PGVector", FakePGVector)

    store.delete_chunks(["a", "b"], connection="postgresql://test")

    assert FakePGVector.last_delete_ids == ["a", "b"]


def test_delete_chunks_is_a_noop_for_no_ids(monkeypatch):
    monkeypatch.setattr(store, "PGVector", FakePGVector)
    FakePGVector.last_delete_ids = "untouched"

    store.delete_chunks([], connection="postgresql://test")

    assert FakePGVector.last_delete_ids == "untouched"
