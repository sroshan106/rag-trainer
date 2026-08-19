"""Phase 8: benchmark eval against tests/benchmark/data/*.csv.

Runs the live graph (needs db + ollama up, vectorstore ingested) and
compares actual retrieval/answers against the labeled question sets.

Questions are dispatched concurrently -- Ollama serves parallel requests, and
the pipeline is entirely I/O-bound from the client's side. Completed answers
are cached to disk as they arrive, so an interrupted run resumes instead of
replaying every LLM call. The cache key includes the grading thresholds, so
re-running after a threshold change re-evaluates rather than reusing stale
answers.

Run: python -m tests.benchmark.run_benchmark [--workers N] [--no-cache]
"""

import argparse
import csv
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.rag.graph import graph
from src.rag.nodes import RETRIEVE_K
from tests.benchmark.cache import CACHE_DIR, ResultCache, config_fingerprint

DATA_DIR = Path(__file__).parent / "data"

# Measured on a 4GB card with llama3.2:3b: 8 concurrent requests finished a
# sample batch in 12.9s against 22.1s at 4. Raise only alongside VRAM -- past
# the point where Ollama runs out of parallel slots, requests just queue.
DEFAULT_WORKERS = 8

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


def _load_csv(name: str) -> list[dict]:
    with open(DATA_DIR / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _run(query: str) -> dict:
    """Invoke the graph, reduced to the JSON-serialisable fields eval needs."""
    result = graph.invoke({"query": query})
    return {
        "answer": result["answer"],
        "retrieved_indices": [
            str(d.metadata.get("row_index")) for d in result.get("retrieved_docs", [])
        ],
    }


def _run_suite(name: str, rows: list[dict], workers: int, cache: ResultCache) -> list[dict]:
    """Return one result dict per row, in row order, reusing cached answers."""

    def resolve(row: dict) -> dict:
        question = row["question"]
        hit = cache.get(name, question)
        if hit is not None:
            return hit
        result = _run(question)
        cache.put(name, question, result)
        return result

    reused = sum(1 for r in rows if cache.get(name, r["question"]) is not None)
    if reused:
        print(f"  {name}: reusing {reused}/{len(rows)} cached answers")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(resolve, rows))


def eval_answerable(
    name: str,
    workers: int = DEFAULT_WORKERS,
    cache: ResultCache | None = None,
    overlap_threshold: float = 0.3,
) -> dict:
    rows = _load_csv(name)
    if not rows:
        return {"name": name, "n": 0}

    cache = cache or ResultCache(CACHE_DIR / "unused.jsonl", enabled=False)
    results = _run_suite(name, rows, workers, cache)

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
        "n": len(rows),
        f"recall@{RETRIEVE_K}": statistics.mean(recalls),
        "mean_answer_overlap": statistics.mean(overlaps),
        f"pass_rate(overlap>={overlap_threshold})": pass_rate,
    }


def eval_no_answer(
    name: str,
    workers: int = DEFAULT_WORKERS,
    cache: ResultCache | None = None,
) -> dict:
    rows = _load_csv(name)
    if not rows:
        return {"name": name, "n": 0}

    cache = cache or ResultCache(CACHE_DIR / "unused.jsonl", enabled=False)
    results = _run_suite(name, rows, workers, cache)

    refused = sum(
        1
        for result in results
        if any(p in result["answer"].lower() for p in REFUSAL_PATTERNS)
    )
    return {
        "name": name,
        "n": len(rows),
        "correct_refusal_rate": refused / len(rows),
    }


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
        "--no-cache",
        action="store_true",
        help="ignore and do not write cached answers",
    )
    args = parser.parse_args(argv)

    fingerprint = config_fingerprint()
    cache_path = CACHE_DIR / f"{fingerprint}.jsonl"
    print(f"config {fingerprint}, {args.workers} workers")
    if not args.no_cache:
        print(f"cache {cache_path}")

    with ResultCache(cache_path, enabled=not args.no_cache) as cache:
        try:
            results = [
                eval_answerable("single_passage_answer_questions.csv", args.workers, cache),
                eval_answerable("multi_passage_answer_questions.csv", args.workers, cache),
                eval_no_answer("no_answer_questions.csv", args.workers, cache),
            ]
        except KeyboardInterrupt:
            # Every answer completed before the interrupt is already flushed;
            # re-running picks up from there.
            print(f"\ninterrupted -- {len(cache)} answers cached, re-run to resume")
            return 130

    _report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
