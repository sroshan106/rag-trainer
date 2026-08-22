from tests.benchmark import run_benchmark
from tests.benchmark.cache import ResultCache


def test_roundtrip_survives_reopen(tmp_path):
    path = tmp_path / "c.jsonl"
    with ResultCache(path) as cache:
        cache.put("suite", "q", {"answer": "a", "retrieved_indices": ["1"]})

    with ResultCache(path) as reopened:
        assert reopened.get("suite", "q")["answer"] == "a"


def test_empty_cache_is_not_falsy(tmp_path):
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
    path = tmp_path / "c.jsonl"
    with ResultCache(path) as cache:
        cache.put("suite", "good", {"answer": "a"})
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"suite": "bad", "quest')

    with ResultCache(path) as reopened:
        assert reopened.get("suite", "good") is not None
        assert len(reopened) == 1


def test_eval_uses_the_cache_it_is_given(tmp_path, monkeypatch):
    test_csv = tmp_path / "test.csv"
    test_csv.write_text("question,answer\nq1,a1\nq2,a2\nq3,a3\n")
    monkeypatch.setattr(
        run_benchmark, "_run", lambda q, model=None: {"answer": "x", "retrieved_indices": ["7"]}
    )
    suite = run_benchmark.build_suite(test_csv, sample=2)
    with ResultCache(tmp_path / "c.jsonl") as cache:
        run_benchmark._run_suite(suite.name, suite.rows, 2, cache, None)
        assert len(cache) == 2


def test_sampling_is_stable_across_calls(tmp_path):
    test_csv = tmp_path / "test.csv"
    rows = "\n".join([f"question {i},answer {i}" for i in range(20)])
    test_csv.write_text(f"question,answer\n{rows}\n")
    a = run_benchmark._load_csv(test_csv, 5)
    b = run_benchmark._load_csv(test_csv, 5)
    assert [r["question"] for r in a] == [r["question"] for r in b]
    assert len(a) == 5
