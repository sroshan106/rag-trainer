from langchain_core.documents import Document

from src.rag.citations import (
    citations_enabled,
    collect_sources,
    format_sources,
)


def test_citations_enabled_by_default(monkeypatch):
    monkeypatch.delenv("RAG_CITATIONS", raising=False)

    assert citations_enabled() is True


def test_citations_disabled_by_falsy_values(monkeypatch):
    for value in ("false", "FALSE", "0", "no", "off", " off "):
        monkeypatch.setenv("RAG_CITATIONS", value)

        assert citations_enabled() is False, value


def test_citations_enabled_by_truthy_value(monkeypatch):
    monkeypatch.setenv("RAG_CITATIONS", "true")

    assert citations_enabled() is True


def test_collect_sources_dedupes_and_keeps_order():
    docs = [
        Document(page_content="a", metadata={"source": "b.txt"}),
        Document(page_content="b", metadata={"source": "a.txt"}),
        Document(page_content="c", metadata={"source": "b.txt"}),
    ]

    assert collect_sources(docs) == ["b.txt", "a.txt"]


def test_collect_sources_defaults_missing_and_null_source():
    docs = [
        Document(page_content="a", metadata={}),
        Document(page_content="b", metadata={"source": None}),
    ]

    assert collect_sources(docs) == ["unknown"]


def test_format_sources_numbers_entries():
    result = format_sources(["a.txt", "b.txt"])

    assert result == "sources:\n  [1] a.txt\n  [2] b.txt"


def test_format_sources_empty_when_no_sources():
    assert format_sources([]) == ""
