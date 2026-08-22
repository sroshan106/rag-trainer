import pytest
from langchain_core.documents import Document

from src.vectorstore import store


class FakePGVector:
    last_init_kwargs = None
    last_delete_ids = None
    added = []

    def __init__(self, **kwargs):
        FakePGVector.last_init_kwargs = kwargs

    def add_embeddings(self, texts, embeddings, metadatas=None, ids=None, **kwargs):
        FakePGVector.added.append(
            {"texts": list(texts), "embeddings": embeddings, "metadatas": metadatas, "ids": ids}
        )
        return ids

    def delete(self, ids=None, **kwargs):
        FakePGVector.last_delete_ids = ids


class FakeEmbeddings:

    def __init__(self, fail_times=0):
        self.batches = []
        self.fail_times = fail_times

    def embed_documents(self, texts):
        self.batches.append(list(texts))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("connection reset by peer")
        return [[float(len(text))] for text in texts]


@pytest.fixture
def fake_store(monkeypatch):
    monkeypatch.setattr(store, "PGVector", FakePGVector)
    monkeypatch.setattr(store.time, "sleep", lambda _seconds: None)
    FakePGVector.added = []
    embeddings = FakeEmbeddings()
    monkeypatch.setattr(store, "_embeddings", lambda **_kwargs: embeddings)
    return embeddings


def _all_ids():
    return [chunk_id for call in FakePGVector.added for chunk_id in call["ids"]]


def test_build_vectorstore_passes_collection_and_connection(fake_store):
    chunks = [Document(page_content="x", metadata={})]

    store.build_vectorstore(chunks, connection="postgresql://test")

    kwargs = FakePGVector.last_init_kwargs
    assert kwargs["collection_name"] == store.COLLECTION_NAME
    assert kwargs["connection"] == "postgresql://test"


def test_build_vectorstore_falls_back_to_env(fake_store, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://from-env")

    chunks = [Document(page_content="x", metadata={})]
    store.build_vectorstore(chunks, connection=None)

    assert FakePGVector.last_init_kwargs["connection"] == "postgresql://from-env"


def test_build_vectorstore_generates_and_returns_ids(fake_store):
    chunks = [Document(page_content="x", metadata={}), Document(page_content="y", metadata={})]

    ids = store.build_vectorstore(chunks, connection="postgresql://test")

    assert len(ids) == 2
    assert len(set(ids)) == 2
    assert _all_ids() == ids


def test_build_vectorstore_uses_provided_ids(fake_store):
    chunks = [Document(page_content="x", metadata={})]

    ids = store.build_vectorstore(chunks, connection="postgresql://test", ids=["fixed-id"])

    assert ids == ["fixed-id"]
    assert _all_ids() == ["fixed-id"]


def test_build_vectorstore_embeds_and_writes_in_batches(fake_store):
    chunks = [Document(page_content=str(n), metadata={"n": n}) for n in range(5)]

    ids = store.build_vectorstore(chunks, connection="postgresql://test", batch_size=2)

    assert sorted(len(batch) for batch in fake_store.batches) == [1, 2, 2]
    assert [len(call["ids"]) for call in FakePGVector.added] == [2, 2, 1]
    assert _all_ids() == ids
    assert FakePGVector.added[0]["metadatas"] == [{"n": 0}, {"n": 1}]


def test_build_vectorstore_embeds_duplicate_text_once(fake_store):
    chunks = [
        Document(page_content="same", metadata={"n": 0}),
        Document(page_content="other", metadata={"n": 1}),
        Document(page_content="same", metadata={"n": 2}),
    ]

    ids = store.build_vectorstore(chunks, connection="postgresql://test", batch_size=10)

    assert fake_store.batches == [["same", "other"]]
    assert sorted(_all_ids()) == sorted(ids)
    stored = {call_id: text for call in FakePGVector.added
              for call_id, text in zip(call["ids"], call["texts"])}
    assert stored[ids[0]] == "same" and stored[ids[2]] == "same"
    metadatas = [metadata for call in FakePGVector.added for metadata in call["metadatas"]]
    assert sorted(metadata["n"] for metadata in metadatas) == [0, 1, 2]


def test_build_vectorstore_reuses_the_same_vector_for_duplicates(fake_store):
    chunks = [Document(page_content="same", metadata={}) for _ in range(2)]

    store.build_vectorstore(chunks, connection="postgresql://test")

    vectors = [vector for call in FakePGVector.added for vector in call["embeddings"]]
    assert len(vectors) == 2
    assert vectors[0] == vectors[1]


def test_build_vectorstore_counts_duplicates_in_progress(fake_store):
    chunks = [Document(page_content="same", metadata={}) for _ in range(4)]
    seen = []

    store.build_vectorstore(
        chunks,
        connection="postgresql://test",
        progress=lambda fraction, message: seen.append((fraction, message)),
    )

    assert seen[0] == (0.0, "embedding 1 unique of 4 chunks")
    assert seen[-1] == (1.0, "embedded 4/4 chunks")


def test_build_vectorstore_reports_progress_per_batch(fake_store):
    chunks = [Document(page_content=str(n), metadata={}) for n in range(4)]
    seen = []

    store.build_vectorstore(
        chunks,
        connection="postgresql://test",
        batch_size=2,
        progress=lambda fraction, message: seen.append((fraction, message)),
    )

    assert [fraction for fraction, _message in seen] == [0.5, 1.0]
    assert seen[-1][1] == "embedded 4/4 chunks"


def test_build_vectorstore_skips_work_for_no_chunks(fake_store):
    assert store.build_vectorstore([], connection="postgresql://test") == []
    assert FakePGVector.added == []


def test_embed_batch_retries_a_dropped_connection(monkeypatch):
    monkeypatch.setattr(store.time, "sleep", lambda _seconds: None)
    embeddings = FakeEmbeddings(fail_times=2)

    vectors = store._embed_batch(embeddings, ["a", "b"], retries=3)

    assert len(vectors) == 2
    assert len(embeddings.batches) == 3


def test_embed_batch_halves_a_batch_that_keeps_failing(monkeypatch):
    monkeypatch.setattr(store.time, "sleep", lambda _seconds: None)
    embeddings = FakeEmbeddings(fail_times=2)

    vectors = store._embed_batch(embeddings, ["a", "b"], retries=2)

    assert len(vectors) == 2
    assert embeddings.batches[:2] == [["a", "b"], ["a", "b"]]
    assert embeddings.batches[2:] == [["a"], ["b"]]


def test_embed_batch_raises_when_a_single_chunk_will_not_embed(monkeypatch):
    monkeypatch.setattr(store.time, "sleep", lambda _seconds: None)
    embeddings = FakeEmbeddings(fail_times=99)

    with pytest.raises(RuntimeError, match="connection reset"):
        store._embed_batch(embeddings, ["a"], retries=2)


def test_delete_chunks_deletes_by_id(fake_store):
    store.delete_chunks(["a", "b"], connection="postgresql://test")

    assert FakePGVector.last_delete_ids == ["a", "b"]


def test_delete_chunks_is_a_noop_for_no_ids(fake_store):
    FakePGVector.last_delete_ids = "untouched"

    store.delete_chunks([], connection="postgresql://test")

    assert FakePGVector.last_delete_ids == "untouched"
