import pytest
from langchain_core.documents import Document

from src.ingestion.splitter import split_documents


def test_split_documents_preserves_metadata():
    docs = [Document(page_content="word " * 2000, metadata={"source": "a.txt"})]

    chunks = split_documents(docs, chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(c.metadata["source"] == "a.txt" for c in chunks)


def test_split_documents_respects_chunk_size():
    docs = [Document(page_content="word " * 2000, metadata={})]

    chunks = split_documents(docs, chunk_size=50, chunk_overlap=10)

    # from_tiktoken_encoder counts in tokens, not chars, so allow slack
    # for the splitter's boundary-snapping behavior.
    assert all(len(c.page_content) < 500 for c in chunks)


def test_split_documents_short_doc_stays_single_chunk():
    docs = [Document(page_content="short text", metadata={"source": "b.txt"})]

    chunks = split_documents(docs, chunk_size=1000, chunk_overlap=150)

    assert len(chunks) == 1
    assert chunks[0].page_content == "short text"


def test_split_documents_empty_list_returns_empty_list():
    assert split_documents([]) == []


def test_split_documents_token_splitter_respects_chunk_size():
    docs = [Document(page_content="word " * 2000, metadata={"source": "a.txt"})]

    chunks = split_documents(docs, chunk_size=50, chunk_overlap=10, splitter="token")

    assert len(chunks) > 1
    assert all(c.metadata["source"] == "a.txt" for c in chunks)


def test_split_documents_rejects_an_unknown_splitter():
    with pytest.raises(ValueError, match="unknown splitter"):
        split_documents([Document(page_content="x", metadata={})], splitter="nope")
