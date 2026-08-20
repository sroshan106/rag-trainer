from langchain_core.documents import Document

from src.rag.prompts import RAG_PROMPT, format_context


def test_format_context_joins_the_chunks():
    docs = [
        Document(page_content="fact one", metadata={"filename": "a.txt"}),
        Document(page_content="fact two", metadata={"filename": "b.txt"}),
    ]

    result = format_context(docs)

    assert result == "fact one\n\nfact two"


def test_format_context_does_not_leak_provenance_into_the_prompt():
    """Each chunk used to be prefixed with a [source: ...] header.

    It spent prompt budget on every query and invited the model to name a file
    in its prose despite being told not to; citations are built from metadata
    beside the answer instead.
    """
    docs = [Document(page_content="fact", metadata={"filename": "a.txt", "unit_index": 3})]

    result = format_context(docs)

    assert result == "fact"
    assert "a.txt" not in result


def test_format_context_of_nothing_is_empty():
    assert format_context([]) == ""


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
