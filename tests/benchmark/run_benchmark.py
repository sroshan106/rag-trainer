"""Phase 8: benchmark eval against tests/benchmark/data/*.csv.

Runs the live graph (needs db + ollama up, vectorstore ingested) and
compares actual retrieval/answers against the labeled question sets.

Questions are dispatched concurrently -- Ollama serves parallel requests, and
the pipeline is entirely I/O-bound from the client's side. Completed answers
are cached to disk as they arrive, so an interrupted run resumes instead of
replaying every LLM call. The cache key includes the grading thresholds and
the model, so re-running after a threshold change re-evaluates rather than
reusing stale answers.

The suites are interleaved in chunks rather than run one after another: a
sequential pass leaves the last suite's metrics undefined until the whole run
is nearly over, which makes a partial run useless for judging a config change.
Round-robin chunks of CHUNK_SIZE keep every suite's numbers advancing together,
so stopping early still leaves three comparable (if noisier) metrics.

Run: python -m tests.benchmark.run_benchmark [--workers N] [--sample N]
                                             [--chunk-size N] [--model NAME]
                                             [--no-cache]
"""

import argparse
import csv
import random
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from src.rag.graph import graph
from src.rag.nodes import AVAILABLE_MODELS, RETRIEVE_K
from tests.benchmark.cache import CACHE_DIR, ResultCache, config_fingerprint

DATA_DIR = Path(__file__).parent / "data"

# Questions taken from each suite per round before moving to the next suite.
# Small enough that all three suites report numbers within the first minute of
# a run; large enough that the ThreadPoolExecutor is not rebuilt every couple
# of questions.
CHUNK_SIZE = 10

# Measured on a 4GB card with llama3.2:3b, using real benchmark questions
# (~3.5k-token prompts): 5.3s/question at 1 worker, 2.2s at 2, 2.3s at 4. Gains
# stop at 2 because each parallel slot needs its own KV cache and the card has
# room for very few at this prompt size -- at 8 the run thrashed and made no
# progress at all. Short prompts scale much further; do not tune this on them.
DEFAULT_WORKERS = 4

# Sampling seed. Fixed so --sample draws the same subset every run: two
# configurations are only comparable if they answered the same questions, and
# a resampled subset would move the metric on its own.
SAMPLE_SEED = 20250819

REFUSAL_PATTERNS = (
    "don't have enough context",
    "doesn't contain",
    "does not contain",
    "cannot answer",
    "can't answer",
    "don't know",
    "not mentioned",
    "no information",
)

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "to",
    "and", "or", "for", "with", "by", "at", "from", "as", "that", "this",
    "it", "its", "be", "do", "does", "did", "has", "have", "had",
}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def _answer_overlap(expected: str, actual: str) -> float:
    exp_kw = _keywords(expected)
    if not exp_kw:
        return 0.0
    act_kw = _keywords(actual)
    return len(exp_kw & act_kw) / len(exp_kw)


def _load_csv(name: str, sample: int | None = None) -> list[dict]:
    with open(DATA_DIR / name, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if sample is None or sample >= len(rows):
        return rows
    # Sort the sampled indices so cached rows stay in a stable order across runs.
    picked = sorted(random.Random(SAMPLE_SEED).sample(range(len(rows)), sample))
    return [rows[i] for i in picked]


def _run(query: str, model: str) -> dict:
    """Invoke the graph, reduced to the JSON-serialisable fields eval needs."""
    # No fallback -- every caller down to run_all's own validation must have
    # already resolved a real model before this runs.
    result = graph.invoke({"query": query, "model": model})
    return {
        "answer": result["answer"],
        "retrieved_indices": [
            str(d.metadata.get("row_index")) for d in result.get("retrieved_docs", [])
        ],
    }


def _dummy_cache() -> ResultCache:
    # ``is None``, not truthiness: an empty ResultCache has len() == 0 and is
    # therefore falsy, which would silently swap a real cache for the dummy.
    return ResultCache(CACHE_DIR / "unused.jsonl", enabled=False)


def _run_suite(
    name: str,
    rows: list[dict],
    workers: int,
    cache: ResultCache,
    model: str | None = None,
) -> list[dict]:
    """Return one result dict per row, in row order, reusing cached answers."""

    def resolve(row: dict) -> dict:
        question = row["question"]
        hit = cache.get(name, question)
        if hit is not None:
            return hit
        result = _run(question, model)
        cache.put(name, question, result)
        return result

    reused = sum(1 for r in rows if cache.get(name, r["question"]) is not None)
    if reused:
        print(f"  {name}: reusing {reused}/{len(rows)} cached answers")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(resolve, rows))


def score_answerable(
    name: str, rows: list[dict], results: list[dict], overlap_threshold: float = 0.3
) -> dict:
    """Metrics for however many rows have been answered so far.

    Pure -- no LLM, no cache. Split out from ``eval_answerable`` so the chunked
    scheduler can rescore after every chunk without re-running anything.
    """
    if not results:
        return {"name": name, "n": 0}

    recalls = [
        1.0 if row["document_index"] in set(result["retrieved_indices"]) else 0.0
        for row, result in zip(rows, results)
    ]
    overlaps = [
        _answer_overlap(row["answer"], result["answer"])
        for row, result in zip(rows, results)
    ]

    pass_rate = sum(1 for o in overlaps if o >= overlap_threshold) / len(overlaps)
    return {
        "name": name,
        "n": len(results),
        f"recall@{RETRIEVE_K}": statistics.mean(recalls),
        "mean_answer_overlap": statistics.mean(overlaps),
        f"pass_rate(overlap>={overlap_threshold})": pass_rate,
    }


def score_no_answer(name: str, rows: list[dict], results: list[dict]) -> dict:
    """Refusal rate over however many rows have been answered so far."""
    if not results:
        return {"name": name, "n": 0}

    refused = sum(
        1
        for result in results
        if any(p in result["answer"].lower() for p in REFUSAL_PATTERNS)
    )
    return {
        "name": name,
        "n": len(results),
        "correct_refusal_rate": refused / len(results),
    }


def eval_answerable(
    name: str,
    workers: int = DEFAULT_WORKERS,
    cache: ResultCache | None = None,
    sample: int | None = None,
    overlap_threshold: float = 0.3,
    model: str | None = None,
) -> dict:
    rows = _load_csv(name, sample)
    if not rows:
        return {"name": name, "n": 0}
    # ``is None``, not truthiness: an empty ResultCache has len() == 0 and is
    # therefore falsy, which would silently swap a real cache for the dummy.
    results = _run_suite(
        name, rows, workers, _dummy_cache() if cache is None else cache, model
    )
    return score_answerable(name, rows, results, overlap_threshold)


def eval_no_answer(
    name: str,
    workers: int = DEFAULT_WORKERS,
    cache: ResultCache | None = None,
    sample: int | None = None,
    model: str | None = None,
) -> dict:
    rows = _load_csv(name, sample)
    if not rows:
        return {"name": name, "n": 0}
    # ``is None``, not truthiness: an empty ResultCache has len() == 0 and is
    # therefore falsy, which would silently swap a real cache for the dummy.
    results = _run_suite(
        name, rows, workers, _dummy_cache() if cache is None else cache, model
    )
    return score_no_answer(name, rows, results)


class _Suite:
    """One question set plus the answers collected for it so far."""

    def __init__(self, csv_name: str, scorer: Callable[[str, list, list], dict], rows: list[dict]):
        self.name = csv_name
        self.scorer = scorer
        self.rows = rows
        self.results: list[dict] = []

    @property
    def remaining(self) -> list[dict]:
        return self.rows[len(self.results):]

    def score(self) -> dict:
        return self.scorer(self.name, self.rows, self.results)


def run_all(
    workers: int = DEFAULT_WORKERS,
    sample: int | None = None,
    use_cache: bool = True,
    model: str | None = None,
    chunk_size: int = CHUNK_SIZE,
    on_progress: Callable[[list[dict], int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[dict]:
    """Run all three suites interleaved in chunks and return their metric dicts.

    The programmatic entry point -- ``main`` is a thin CLI wrapper over this so
    the API can run a benchmark without going through argv or stdout.

    ``on_progress(metrics, done, total)`` fires after every chunk with metrics
    for *all* suites scored over what has been answered so far, which is what
    lets a caller display live numbers. ``should_stop()`` is polled at the same
    points; returning True ends the run and returns the partial metrics rather
    than raising, so a stopped run is still a readable result.
    """
    if model not in AVAILABLE_MODELS:
        raise ValueError(
            f"model is required -- choose from {list(AVAILABLE_MODELS)}"
        )

    suites = [
        _Suite("single_passage_answer_questions.csv", score_answerable, _load_csv("single_passage_answer_questions.csv", sample)),
        _Suite("multi_passage_answer_questions.csv", score_answerable, _load_csv("multi_passage_answer_questions.csv", sample)),
        _Suite("no_answer_questions.csv", score_no_answer, _load_csv("no_answer_questions.csv", sample)),
    ]
    total = sum(len(s.rows) for s in suites)

    cache_path = CACHE_DIR / f"{config_fingerprint(model)}.jsonl"
    with ResultCache(cache_path, enabled=use_cache) as cache:
        while any(s.remaining for s in suites):
            for suite in suites:
                if should_stop is not None and should_stop():
                    return [s.score() for s in suites]
                chunk = suite.remaining[:chunk_size]
                if not chunk:
                    continue
                suite.results.extend(_run_suite(suite.name, chunk, workers, cache, model))
                done = sum(len(s.results) for s in suites)
                if on_progress is not None:
                    on_progress([s.score() for s in suites], done, total)

    return [s.score() for s in suites]


def _report(results: list[dict]) -> None:
    for r in results:
        print(f"\n== {r['name']} (n={r['n']}) ==")
        for k, v in r.items():
            if k in ("name", "n"):
                continue
            print(f"  {k}: {v:.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="evaluate N questions per suite instead of all (fixed seed)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="questions taken per suite per round (suites are interleaved)",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(AVAILABLE_MODELS),
        help="model to benchmark -- must already be pulled",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore and do not write cached answers",
    )
    args = parser.parse_args(argv)

    fingerprint = config_fingerprint(args.model)
    scope = f"sample {args.sample}/suite" if args.sample else "full"
    print(
        f"config {fingerprint}, model {args.model}, "
        f"{args.workers} workers, {scope}, chunks of {args.chunk_size}"
    )
    if not args.no_cache:
        print(f"cache {CACHE_DIR / f'{fingerprint}.jsonl'}")

    def _tick(_metrics: list[dict], done: int, total: int) -> None:
        print(f"  {done}/{total} questions answered")

    try:
        results = run_all(
            args.workers,
            args.sample,
            use_cache=not args.no_cache,
            model=args.model,
            chunk_size=args.chunk_size,
            on_progress=_tick,
        )
    except KeyboardInterrupt:
        # Every answer completed before the interrupt is already flushed;
        # re-running picks up from there.
        print("\ninterrupted -- completed answers cached, re-run to resume")
        return 130

    _report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
