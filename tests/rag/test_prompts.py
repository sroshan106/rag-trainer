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


def test_format_context_defaults_null_source_to_unknown():
    docs = [Document(page_content="fact", metadata={"source": None})]

    result = format_context(docs)

    assert "[source: unknown]\nfact" in result


def test_rag_prompt_grounds_and_allows_refusal():
    filled = RAG_PROMPT.format(context="ctx", question="q?")

    assert "ONLY the context" in filled
    assert "don't guess" in filled
    assert "ctx" in filled
    assert "q?" in filled


def test_rag_prompt_leaves_citation_to_the_code_path():
    # The prompt used to ask for inline source tags; the model wasn't reliable
    # at producing them, so citations moved to src/rag/citations.py, which
    # collects them from the documents that actually reached the prompt. The
    # instruction now says the opposite -- assert that, so the two can't drift
    # back into contradicting each other.
    filled = RAG_PROMPT.format(context="ctx", question="q?")

    assert "don't cite or name sources inline" in filled
