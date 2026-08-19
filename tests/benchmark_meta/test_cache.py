"""Tests for the benchmark result cache itself.

The cache is what makes an interrupted benchmark resumable, and a cache that
silently does nothing looks identical to a working one from the outside -- so
its behaviour is asserted rather than assumed.
"""

from tests.benchmark import run_benchmark
from tests.benchmark.cache import ResultCache


def test_roundtrip_survives_reopen(tmp_path):
    path = tmp_path / "c.jsonl"
    with ResultCache(path) as cache:
        cache.put("suite", "q", {"answer": "a", "retrieved_indices": ["1"]})

    with ResultCache(path) as reopened:
        assert reopened.get("suite", "q")["answer"] == "a"


def test_empty_cache_is_not_falsy(tmp_path):
    """Regression: __len__ made a fresh cache falsy, so `cache or fallback`
    swapped the real cache for a disabled one and nothing was ever written."""
    with ResultCache(tmp_path / "c.jsonl") as cache:
        assert cache is not None
        assert len(cache) == 0
        assert (cache if cache is not None else "fallback") is cache


def test_disabled_cache_reads_and_writes_nothing(tmp_path):
    path = tmp_path / "c.jsonl"
    with ResultCache(path, enabled=False) as cache:
        cache.put("suite", "q", {"answer": "a"})
        assert cache.get("suite", "q") is None
    assert not path.exists()


def test_truncated_final_line_is_skipped(tmp_path):
    """A run killed mid-write can leave a partial JSON line."""
    path = tmp_path / "c.jsonl"
    with ResultCache(path) as cache:
        cache.put("suite", "good", {"answer": "a"})
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"suite": "bad", "quest')

    with ResultCache(path) as reopened:
        assert reopened.get("suite", "good") is not None
        assert len(reopened) == 1


def test_eval_uses_the_cache_it_is_given(tmp_path, monkeypatch):
    """The whole point of passing a cache in: answers must land in it."""
    monkeypatch.setattr(
        run_benchmark, "_run", lambda q: {"answer": "x", "retrieved_indices": ["7"]}
    )
    with ResultCache(tmp_path / "c.jsonl") as cache:
        run_benchmark.eval_answerable(
            "single_passage_answer_questions.csv", workers=2, cache=cache, sample=2
        )
        assert len(cache) == 2


def test_sampling_is_stable_across_calls():
    a = run_benchmark._load_csv("single_passage_answer_questions.csv", 5)
    b = run_benchmark._load_csv("single_passage_answer_questions.csv", 5)
    assert [r["question"] for r in a] == [r["question"] for r in b]
    assert len(a) == 5
