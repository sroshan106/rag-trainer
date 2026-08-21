"""Benchmark eval against test suites or custom test files.

Runs the live graph (needs db + ollama up, vectorstore ingested) and
compares actual retrieval/answers against the labeled question sets.
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

from src.benchmark.cache import CACHE_DIR, ResultCache, config_fingerprint
from src.rag.graph import graph
from src.rag.nodes import AVAILABLE_MODELS, RETRIEVE_K

CHUNK_SIZE = 10
DEFAULT_WORKERS = 4
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


def _normalize_csv_row(row: dict, mapping: dict | None = None) -> dict | None:
    """Extract standard question, answer, and document_index from a CSV row."""
    question = None
    answer = None
    doc_index = None

    if mapping:
        q_col = mapping.get("question_col")
        if q_col and q_col in row and row[q_col] and str(row[q_col]).strip():
            question = str(row[q_col]).strip()
        ans_col = mapping.get("answer_col")
        if ans_col and ans_col in row and row[ans_col] is not None:
            answer = str(row[ans_col]).strip()
        d_col = mapping.get("doc_index_col")
        if d_col and d_col in row and row[d_col] is not None and str(row[d_col]).strip() != "":
            doc_index = str(row[d_col]).strip()

    clean_row = {k.strip().lower(): v for k, v in row.items() if k is not None}

    if not question:
        for key in ("question", "query", "prompt", "q", "question_text"):
            if key in clean_row and clean_row[key] and clean_row[key].strip():
                question = clean_row[key].strip()
                break

    if not question:
        return None

    if answer is None:
        for key in ("answer", "ground_truth", "reference", "expected", "target", "a", "expected_answer"):
            if key in clean_row and clean_row[key] is not None:
                answer = clean_row[key].strip()
                break

    if doc_index is None:
        for key in ("document_index", "doc_index", "doc_id", "index", "document_id"):
            if key in clean_row and clean_row[key] is not None and clean_row[key].strip() != "":
                doc_index = clean_row[key].strip()
                break

    res = {"question": question}
    if answer is not None:
        res["answer"] = answer
    if doc_index is not None:
        res["document_index"] = doc_index
    return res


def _resolve_csv_path(path_or_name: str | Path) -> Path:
    p = Path(path_or_name)
    if p.is_file():
        return p
    upload_p = Path("data/benchmark_uploads") / p
    if upload_p.is_file():
        return upload_p
    try:
        from src.benchmark.files import resolve_test_file_path
        return resolve_test_file_path(str(path_or_name))
    except Exception:
        pass
    return p


def _load_csv(
    name: str | Path, sample: int | None = None, mapping: dict | None = None
) -> list[dict]:
    resolved = _resolve_csv_path(name)
    if mapping is None:
        try:
            from src.benchmark.files import get_test_file_entry
            entry = get_test_file_entry(str(name)) or get_test_file_entry(str(resolved))
            if entry:
                mapping = {
                    "question_col": entry.get("question_col"),
                    "answer_col": entry.get("answer_col"),
                    "doc_index_col": entry.get("doc_index_col"),
                }
        except Exception:
            pass

    with open(resolved, newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    rows = []
    for r in raw_rows:
        normalized = _normalize_csv_row(r, mapping=mapping)
        if normalized is not None:
            rows.append(normalized)

    if sample is None or sample >= len(rows):
        return rows
    picked = sorted(random.Random(SAMPLE_SEED).sample(range(len(rows)), sample))
    return [rows[i] for i in picked]


def _run(query: str, model: str) -> dict:
    """Invoke the graph, reduced to the JSON-serialisable fields eval needs."""
    result = graph.invoke({"query": query, "model": model})
    return {
        "answer": result["answer"],
        "retrieved_indices": [
            str(d.metadata.get("row_index")) for d in result.get("retrieved_docs", [])
        ],
    }


def _dummy_cache() -> ResultCache:
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
        current_run = sys.modules[__name__]._run
        result = current_run(question, model)
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
    if not results:
        return {"name": name, "n": 0}

    has_doc_index = any(
        row.get("document_index") is not None and str(row.get("document_index")).strip() != ""
        for row in rows
    )
    recalls = []
    if has_doc_index:
        for row, result in zip(rows, results):
            gold_ids = set(re.findall(r"\d+", str(row.get("document_index", ""))))
            retrieved = set(result.get("retrieved_indices", []))
            recalls.append(1.0 if gold_ids and gold_ids & retrieved else 0.0)

    overlaps = [
        _answer_overlap(row.get("answer", ""), result.get("answer", ""))
        for row, result in zip(rows, results)
    ]

    pass_rate = sum(1 for o in overlaps if o >= overlap_threshold) / len(overlaps) if overlaps else 0.0
    metrics = {
        "name": name,
        "n": len(results),
    }
    if recalls:
        metrics[f"recall@{RETRIEVE_K}"] = statistics.mean(recalls)
    metrics["mean_answer_overlap"] = statistics.mean(overlaps) if overlaps else 0.0
    metrics[f"pass_rate(overlap>={overlap_threshold})"] = pass_rate
    return metrics


def score_no_answer(name: str, rows: list[dict], results: list[dict]) -> dict:
    if not results:
        return {"name": name, "n": 0}

    refused = sum(
        1
        for result in results
        if any(p in result.get("answer", "").lower() for p in REFUSAL_PATTERNS)
    )
    return {
        "name": name,
        "n": len(results),
        "correct_refusal_rate": refused / len(results),
    }


def eval_answerable(
    name: str | Path,
    workers: int = DEFAULT_WORKERS,
    cache: ResultCache | None = None,
    sample: int | None = None,
    overlap_threshold: float = 0.3,
    model: str | None = None,
) -> dict:
    rows = _load_csv(name, sample)
    suite_name = Path(name).name
    if not rows:
        return {"name": suite_name, "n": 0}
    results = _run_suite(
        suite_name, rows, workers, _dummy_cache() if cache is None else cache, model
    )
    return score_answerable(suite_name, rows, results, overlap_threshold)


def eval_no_answer(
    name: str | Path,
    workers: int = DEFAULT_WORKERS,
    cache: ResultCache | None = None,
    sample: int | None = None,
    model: str | None = None,
) -> dict:
    rows = _load_csv(name, sample)
    suite_name = Path(name).name
    if not rows:
        return {"name": suite_name, "n": 0}
    results = _run_suite(
        suite_name, rows, workers, _dummy_cache() if cache is None else cache, model
    )
    return score_no_answer(suite_name, rows, results)


class _Suite:
    def __init__(self, name: str, scorer: Callable[[str, list, list], dict], rows: list[dict]):
        self.name = name
        self.scorer = scorer
        self.rows = rows
        self.results: list[dict] = []

    @property
    def remaining(self) -> list[dict]:
        return self.rows[len(self.results):]

    def score(self) -> dict:
        return self.scorer(self.name, self.rows, self.results)


def build_suite(
    path_or_name: str | Path, sample: int | None = None, name: str | None = None
) -> _Suite:
    resolved = _resolve_csv_path(path_or_name)
    suite_name = name or resolved.name
    rows = _load_csv(resolved, sample=sample)
    has_answers = any(bool(r.get("answer") and str(r.get("answer")).strip()) for r in rows)
    scorer = score_answerable if has_answers else score_no_answer
    return _Suite(suite_name, scorer, rows)


def run_all(
    workers: int = DEFAULT_WORKERS,
    sample: int | None = None,
    use_cache: bool = True,
    model: str | None = None,
    chunk_size: int = CHUNK_SIZE,
    on_progress: Callable[[list[dict], int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    test_files: list[str | Path] | None = None,
) -> list[dict]:
    if model not in AVAILABLE_MODELS:
        raise ValueError(
            f"model is required -- choose from {list(AVAILABLE_MODELS)}"
        )

    if test_files:
        suites = [build_suite(f, sample=sample) for f in test_files]
    else:
        from src.benchmark.files import list_test_files
        available = list_test_files()
        suites = [
            build_suite(s.get("stored_path") or s["id"], sample=sample, name=s.get("name"))
            for s in available
        ]
    total = sum(len(s.rows) for s in suites)

    if total == 0:
        return [s.score() for s in suites]

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
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")


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
    parser.add_argument(
        "--test-file",
        action="append",
        dest="test_files",
        help="custom test CSV file(s) to benchmark (can be specified multiple times)",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        help="directory containing custom test CSV files",
    )
    args = parser.parse_args(argv)

    test_files = args.test_files or []
    if args.test_dir:
        if not args.test_dir.is_dir():
            print(f"error: {args.test_dir} is not a directory", file=sys.stderr)
            return 1
        found_csvs = sorted(args.test_dir.glob("*.csv"))
        test_files.extend(str(p) for p in found_csvs if not p.name.startswith("documents"))

    test_files_arg = test_files if test_files else None

    fingerprint = config_fingerprint(args.model)
    scope = f"sample {args.sample}/suite" if args.sample else "full"
    suites_desc = f"{len(test_files)} custom suites" if test_files else "all configured suites"
    print(
        f"config {fingerprint}, model {args.model}, "
        f"{args.workers} workers, {scope}, chunks of {args.chunk_size} ({suites_desc})"
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
            test_files=test_files_arg,
        )
    except KeyboardInterrupt:
        print("\ninterrupted -- completed answers cached, re-run to resume")
        return 130

    _report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
