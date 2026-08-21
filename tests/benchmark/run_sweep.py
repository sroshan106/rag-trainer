"""Phase 8: grading-threshold sweep.

RELEVANCE_FLOOR and RELEVANCE_RATIO trade answerable-question accuracy against
correct refusal, and picking either one from a single benchmark run measures a
point rather than the curve. This replays a sampled benchmark across a grid of
cutoffs and reports both sides plus their combined score, so the operating
point is argued from the curve.

Each grid point is a distinct cache fingerprint, so results accumulate across
runs and an interrupted sweep resumes.

Run: python -m tests.benchmark.run_sweep --model <model> [--sample N] [--workers N]
"""

import argparse
import importlib
import sys

from tests.benchmark import run_benchmark

# Bracket the observed score distribution: off-topic queries top out around
# 0.45 and on-topic hits start around 0.52, so the useful floors sit between.
FLOORS = (0.44, 0.48, 0.52, 0.56, 0.60)
# Held at one value by default. Since hybrid retrieval keeps every full-text
# hit regardless of cosine, the ratio now only reorders dense-only documents,
# and sweeping it doubles a run that is already dominated by generation time.
RATIOS = (0.90,)

DEFAULT_SAMPLE = 15


def _reconfigure(floor: float, ratio: float) -> None:
    """Rebind the cutoffs the grader reads.

    The constants are module-level and read at import, so the sweep sets them
    directly rather than through the environment -- reloading the module would
    also discard the cached vectorstore and LLM clients between grid points.
    """
    run_benchmark_nodes = importlib.import_module("src.rag.nodes")
    run_benchmark_nodes.RELEVANCE_FLOOR = floor
    run_benchmark_nodes.RELEVANCE_RATIO = ratio


def combined_score(results: list[dict]) -> dict:
    """Collapse all suites into one comparable number.

    Answerable suites contribute their pass rate, no-answer suites their
    correct-refusal rate, each weighted by question count. A cutoff that wins
    only by refusing everything, or only by answering everything, cannot score
    well on this -- which is the whole point of sweeping.
    """
    correct = 0.0
    total = 0
    parts = {}
    for r in results:
        n = r["n"]
        if not n:
            continue
        rate = next(
            (v for k, v in r.items() if k.startswith("pass_rate")),
            r.get("correct_refusal_rate"),
        )
        short = r["name"].replace(".csv", "").replace("_", " ").strip()
        parts[short] = round(rate, 3)
        correct += rate * n
        total += n
    return {"combined": round(correct / total, 3) if total else 0.0, **parts}


def sweep(sample: int, workers: int, model: str) -> list[dict]:
    rows = []
    for ratio in RATIOS:
        for floor in FLOORS:
            _reconfigure(floor, ratio)
            print(f"  floor={floor} ratio={ratio} ...", flush=True)
            results = run_benchmark.run_all(workers=workers, sample=sample, model=model)
            row = {"floor": floor, "ratio": ratio, **combined_score(results)}
            rows.append(row)
            print(f"    -> combined {row['combined']}", flush=True)
    return rows


def _report(rows: list[dict]) -> None:
    keys = [k for k in rows[0] if k not in ("floor", "ratio", "combined")]
    header = ["floor", "ratio", "combined", *keys]
    print("\n| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for row in sorted(rows, key=lambda r: -r["combined"]):
        print("| " + " | ".join(f"{row[h]}" for h in header) + " |")

    best = max(rows, key=lambda r: r["combined"])
    print(f"\nbest: floor={best['floor']} ratio={best['ratio']} combined={best['combined']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=list(run_benchmark.AVAILABLE_MODELS),
        help="model to sweep -- must already be pulled",
    )
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--workers", type=int, default=run_benchmark.DEFAULT_WORKERS)
    args = parser.parse_args(argv)

    print(f"sweeping {len(FLOORS) * len(RATIOS)} points, {args.sample} questions/suite, model {args.model}")
    rows = sweep(args.sample, args.workers, args.model)
    _report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
