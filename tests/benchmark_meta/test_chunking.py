import pytest
from tests.benchmark import run_benchmark


@pytest.fixture
def fake_llm(monkeypatch):
    asked: list[str] = []

    def fake_run(query, model=None):
        asked.append(query)
        return {"answer": "the answer mentions nothing", "retrieved_indices": ["1"]}

    monkeypatch.setattr(run_benchmark, "_run", fake_run)
    return asked


@pytest.fixture
def test_suites(tmp_path):
    s1 = tmp_path / "s1.csv"
    s2 = tmp_path / "s2.csv"
    s3 = tmp_path / "s3.csv"
    s1.write_text("question,answer\n" + "\n".join([f"s1_q{i},ans" for i in range(30)]))
    s2.write_text("question,answer\n" + "\n".join([f"s2_q{i},ans" for i in range(30)]))
    s3.write_text("question,answer\n" + "\n".join([f"s3_q{i}," for i in range(30)]))
    return [str(s1), str(s2), str(s3)]


def test_every_suite_reports_after_the_first_round(fake_llm, test_suites):
    ticks = []
    run_benchmark.run_all(
        workers=2,
        sample=25,
        use_cache=False,
        model="llama3.2:3b",
        chunk_size=10,
        test_files=test_suites,
        on_progress=lambda metrics, done, total: ticks.append((metrics, done, total)),
    )

    metrics, done, total = ticks[2]
    assert done == 30
    assert total == 75
    assert [m["n"] for m in metrics] == [10, 10, 10]


def test_progress_is_monotonic_and_ends_complete(fake_llm, test_suites):
    ticks = []
    results = run_benchmark.run_all(
        workers=2,
        sample=25,
        use_cache=False,
        model="llama3.2:3b",
        chunk_size=10,
        test_files=test_suites,
        on_progress=lambda metrics, done, total: ticks.append(done),
    )

    assert ticks == sorted(ticks)
    assert ticks[-1] == 75
    assert [r["n"] for r in results] == [25, 25, 25]


def test_stopping_returns_partial_metrics_for_all_suites(fake_llm, test_suites):
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 3

    results = run_benchmark.run_all(
        workers=2,
        sample=25,
        use_cache=False,
        model="llama3.2:3b",
        chunk_size=10,
        test_files=test_suites,
        should_stop=should_stop,
    )

    assert [r["n"] for r in results] == [10, 10, 10]
    assert "correct_refusal_rate" in results[2]
    assert len(fake_llm) == 30


def test_stopping_before_the_first_chunk_yields_empty_suites(fake_llm, test_suites):
    results = run_benchmark.run_all(
        workers=2, sample=5, use_cache=False, model="llama3.2:3b", test_files=test_suites, should_stop=lambda: True
    )

    assert [r["n"] for r in results] == [0, 0, 0]
    assert fake_llm == []


def test_unknown_model_is_rejected_before_any_llm_call(fake_llm, test_suites):
    with pytest.raises(ValueError, match="model is required"):
        run_benchmark.run_all(sample=5, use_cache=False, model="gpt-nope", test_files=test_suites)

    assert fake_llm == []
