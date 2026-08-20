"""Tests for the chunked, interleaved benchmark scheduler.

The reason the suites are interleaved at all is that a stopped run must leave
usable numbers for *every* suite -- a sequential pass would leave the last one
at n=0. These tests pin that property, and the stop path that depends on it.
"""

import pytest

from tests.benchmark import run_benchmark


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the graph call with a deterministic stub, recording call order."""
    asked: list[str] = []

    def fake_run(query, model=None):
        asked.append(query)
        return {"answer": "the answer mentions nothing", "retrieved_indices": ["1"]}

    monkeypatch.setattr(run_benchmark, "_run", fake_run)
    return asked


def test_every_suite_reports_after_the_first_round(fake_llm):
    ticks = []
    run_benchmark.run_all(
        workers=2,
        sample=25,
        use_cache=False,
        model="llama3.2:3b",
        chunk_size=10,
        on_progress=lambda metrics, done, total: ticks.append((metrics, done, total)),
    )

    # Three chunks in, one per suite: every suite has answered its first 10.
    metrics, done, total = ticks[2]
    assert done == 30
    assert total == 75
    assert [m["n"] for m in metrics] == [10, 10, 10]


def test_progress_is_monotonic_and_ends_complete(fake_llm):
    ticks = []
    results = run_benchmark.run_all(
        workers=2,
        sample=25,
        use_cache=False,
        model="llama3.2:3b",
        chunk_size=10,
        on_progress=lambda metrics, done, total: ticks.append(done),
    )

    assert ticks == sorted(ticks)
    assert ticks[-1] == 75
    assert [r["n"] for r in results] == [25, 25, 25]


def test_stopping_returns_partial_metrics_for_all_suites(fake_llm):
    calls = {"n": 0}

    def should_stop():
        # Let one chunk per suite through, then stop.
        calls["n"] += 1
        return calls["n"] > 3

    results = run_benchmark.run_all(
        workers=2,
        sample=25,
        use_cache=False,
        model="llama3.2:3b",
        chunk_size=10,
        should_stop=should_stop,
    )

    assert [r["n"] for r in results] == [10, 10, 10]
    # Partial metrics are real metrics, not placeholders.
    assert "correct_refusal_rate" in results[2]
    assert len(fake_llm) == 30


def test_stopping_before_the_first_chunk_yields_empty_suites(fake_llm):
    results = run_benchmark.run_all(
        workers=2, sample=5, use_cache=False, model="llama3.2:3b", should_stop=lambda: True
    )

    assert [r["n"] for r in results] == [0, 0, 0]
    assert fake_llm == []


def test_unknown_model_is_rejected_before_any_llm_call(fake_llm):
    with pytest.raises(ValueError, match="model is required"):
        run_benchmark.run_all(sample=5, use_cache=False, model="gpt-nope")

    assert fake_llm == []
