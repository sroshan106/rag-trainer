"""Phase 8: end-to-end integration tests against live Postgres + Ollama.

Excluded from the default run (see pytest.ini) because they need both services
up and an ingested vectorstore. Run with:

    RAG_INTEGRATION=1 pytest -m integration
"""

import os

import pytest

pytestmark = pytest.mark.integration

if not os.environ.get("RAG_INTEGRATION"):
    pytest.skip("set RAG_INTEGRATION=1 to run", allow_module_level=True)

from src.observability import tracing  # noqa: E402
from src.rag.graph import ask, graph  # noqa: E402
from src.rag.grade import RELEVANCE_FLOOR, SCORE_KEY  # noqa: E402
from src.rag.models import AVAILABLE_MODELS  # noqa: E402

# No default model exists anymore -- these tests exercise the pipeline
# end-to-end and must name one explicitly, same as any real caller would.
TEST_MODEL = AVAILABLE_MODELS[0]

# A fact that is unambiguously in tests/benchmark/data/documents.csv. If the corpus changes,
# this is the assertion that needs updating.
KNOWN_QUERY = "What are Bullet Kin?"
KNOWN_FACT = "enem"  # "enemy"/"enemies" — matches without pinning exact phrasing
KNOWN_SOURCE = "enterthegungeon.fandom.com"

OFF_TOPIC_QUERY = "asdkjh qwe zxc nonsense gibberish"


def _urls(result: dict) -> list[str]:
    """The link each citation points at. See src/rag/citations.py."""
    return [c["url"] for c in result["citations"] if c.get("url")]


def test_known_query_returns_grounded_answer():
    result = ask(KNOWN_QUERY, model=TEST_MODEL)

    assert KNOWN_FACT in result["answer"].lower()
    assert result["citations"], "expected at least one citation"
    assert any(KNOWN_SOURCE in url for url in _urls(result))


def test_known_query_cites_only_relevant_sources():
    # Regression guard: before relevance grading, this query cited unrelated
    # documents about fantasy books and GPT tutorials alongside the real hit.
    result = ask(KNOWN_QUERY, model=TEST_MODEL)

    assert all(KNOWN_SOURCE in url for url in _urls(result))


def test_citations_locate_the_row_they_came_from():
    result = ask(KNOWN_QUERY, model=TEST_MODEL)

    for citation in result["citations"]:
        assert citation["file_id"]
        assert citation["unit_index"] is not None
        assert citation["label"] == f"{citation['unit_kind']} {citation['unit_index']}"


def test_citation_fields_carry_the_source_columns():
    # The corpus declares ``index`` and ``source_url``; both are lifted out as
    # citation fields at ingest, so an answer can name where a row came from
    # rather than only where it sits. See src/ingestion/units.py.
    result = ask(KNOWN_QUERY, model=TEST_MODEL)

    assert result["citations"]
    for citation in result["citations"]:
        assert citation["fields"], "expected citation fields from the CSV columns"
        assert KNOWN_SOURCE in citation["fields"]["source_url"]


def test_off_topic_query_refuses_rather_than_guessing():
    result = ask(OFF_TOPIC_QUERY, model=TEST_MODEL)

    assert "don't have enough context" in result["answer"]
    assert result["citations"] == []


def test_retrieval_returns_k_scored_documents():
    state = graph.invoke({"query": KNOWN_QUERY, "model": TEST_MODEL})

    assert len(state["retrieved_docs"]) > 0
    for doc in state["retrieved_docs"]:
        assert SCORE_KEY in doc.metadata


def test_graded_docs_all_clear_the_floor():
    state = graph.invoke({"query": KNOWN_QUERY, "model": TEST_MODEL})

    assert state["graded_docs"]
    for doc in state["graded_docs"]:
        assert doc.metadata[SCORE_KEY] >= RELEVANCE_FLOOR


def test_tracing_records_a_span_per_node():
    with tracing.collect() as spans:
        ask(KNOWN_QUERY, model=TEST_MODEL)

    names = [s["span"] for s in spans]
    assert names == ["retrieve", "grade", "generate", "ask"]
    assert all(s["duration_ms"] >= 0 for s in spans)


def test_generate_span_reports_ollama_token_counts():
    with tracing.collect() as spans:
        ask(KNOWN_QUERY, model=TEST_MODEL)

    generate = next(s for s in spans if s["span"] == "generate")
    assert generate["eval_count"] > 0
    assert generate["tokens_per_sec"] > 0
