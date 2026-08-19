from langchain_core.documents import Document

from src.rag.prompts import RAG_PROMPT, format_context


def test_format_context_includes_source_metadata():
    docs = [
        Document(page_content="fact one", metadata={"source": "a.txt"}),
        Document(page_content="fact two", metadata={"source": "b.txt"}),
    ]

    result = format_context(docs)

    assert "[source: a.txt]\nfact one" in result
    assert "[source: b.txt]\nfact two" in result


def test_format_context_defaults_to_unknown_source():
    docs = [Document(page_content="fact", metadata={})]

    result = format_context(docs)

    assert "[source: unknown]\nfact" in result


def test_rag_prompt_grounds_and_allows_refusal():
    filled = RAG_PROMPT.format(context="ctx", question="q?")

    assert "ONLY the context" in filled
    assert "don't guess" in filled
    assert "source tag" in filled
    assert "ctx" in filled
    assert "q?" in filled
