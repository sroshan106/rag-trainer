"""Phase 8: benchmark eval against tests/benchmark/data/*.csv.

Runs the live graph (needs db + ollama up, vectorstore ingested) and
compares actual retrieval/answers against the labeled question sets.

Run: python -m tests.benchmark.run_benchmark
"""

import csv
import re
import statistics
from pathlib import Path

from src.rag.graph import graph
from src.rag.nodes import RETRIEVE_K

DATA_DIR = Path(__file__).parent / "data"

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
    return graph.invoke({"query": query})


def eval_answerable(name: str, overlap_threshold: float = 0.3) -> dict:
    rows = _load_csv(name)
    if not rows:
        return {"name": name, "n": 0}

    overlaps = []
    recalls = []
    for row in rows:
        expected_doc_index = row["document_index"]
        result = _run(row["question"])

        retrieved_indices = {
            str(d.metadata.get("row_index")) for d in result.get("retrieved_docs", [])
        }
        recalls.append(1.0 if expected_doc_index in retrieved_indices else 0.0)

        overlaps.append(_answer_overlap(row["answer"], result["answer"]))

    pass_rate = sum(1 for o in overlaps if o >= overlap_threshold) / len(overlaps)
    return {
        "name": name,
        "n": len(rows),
        f"recall@{RETRIEVE_K}": statistics.mean(recalls),
        "mean_answer_overlap": statistics.mean(overlaps),
        f"pass_rate(overlap>={overlap_threshold})": pass_rate,
    }


def eval_no_answer(name: str) -> dict:
    rows = _load_csv(name)
    if not rows:
        return {"name": name, "n": 0}

    refused = 0
    for row in rows:
        result = _run(row["question"])
        answer_lower = result["answer"].lower()
        if any(p in answer_lower for p in REFUSAL_PATTERNS):
            refused += 1
    return {
        "name": name,
        "n": len(rows),
        "correct_refusal_rate": refused / len(rows),
    }


def main() -> None:
    results = [
        eval_answerable("single_passage_answer_questions.csv"),
        eval_answerable("multi_passage_answer_questions.csv"),
        eval_no_answer("no_answer_questions.csv"),
    ]

    for r in results:
        print(f"\n== {r['name']} (n={r['n']}) ==")
        for k, v in r.items():
            if k in ("name", "n"):
                continue
            print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
