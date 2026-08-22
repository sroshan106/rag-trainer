import argparse
import importlib
import sys

from tests.benchmark import run_benchmark

FLOORS = (0.44, 0.48, 0.52, 0.56, 0.60)
RATIOS = (0.90,)

DEFAULT_SAMPLE = 15


def _reconfigure(floor: float, ratio: float) -> None:
    grade = importlib.import_module("src.rag.grade")
    grade.RELEVANCE_FLOOR = floor
    grade.RELEVANCE_RATIO = ratio


def combined_score(results: list[dict]) -> dict:
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
