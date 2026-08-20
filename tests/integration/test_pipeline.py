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
from src.rag.nodes import AVAILABLE_MODELS, RELEVANCE_FLOOR, SCORE_KEY  # noqa: E402

# No default model exists anymore -- these tests exercise the pipeline
# end-to-end and must name one explicitly, same as any real caller would.
TEST_MODEL = AVAILABLE_MODELS[0]

# A fact that is unambiguously in tests/benchmark/data/documents.csv. If the corpus changes,
# this is the assertion that needs updating.
KNOWN_QUERY = "What are Bullet Kin?"
KNOWN_FACT = "enem"  # "enemy"/"enemies" — matches without pinning exact phrasing
KNOWN_SOURCE = "enterthegungeon.fandom.com"

OFF_TOPIC_QUERY = "asdkjh qwe zxc nonsense gibberish"


def test_known_query_returns_grounded_answer():
    result = ask(KNOWN_QUERY, model=TEST_MODEL)

    assert KNOWN_FACT in result["answer"].lower()
    assert result["sources"], "expected at least one source"
    assert any(KNOWN_SOURCE in s for s in result["sources"])


def test_known_query_cites_only_relevant_sources():
    # Regression guard: before relevance grading, this query cited unrelated
    # documents about fantasy books and GPT tutorials alongside the real hit.
    result = ask(KNOWN_QUERY, model=TEST_MODEL)

    assert all(KNOWN_SOURCE in s for s in result["sources"])


def test_off_topic_query_refuses_rather_than_guessing():
    result = ask(OFF_TOPIC_QUERY, model=TEST_MODEL)

    assert "don't have enough context" in result["answer"]
    assert result["sources"] == []


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
