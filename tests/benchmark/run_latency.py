"""Phase 8: end-to-end latency measurement.

Replays queries through the live graph and reports percentiles per node, using
the spans that src/observability/tracing.py already records — there is no
separate timing path.

Run: python -m tests.benchmark.run_latency [repeats]

Needs db + ollama up and an ingested vectorstore. Single-threaded on purpose:
Ollama does serve concurrent requests (run_benchmark.py exploits that), but
under load the per-request numbers measure contention for parallel slots
rather than the pipeline's own cost, which is what this script reports.
"""

import statistics
import sys
from collections import defaultdict

from src.observability import tracing
from src.rag.graph import ask
from src.rag.model_catalog import list_installed

# Mix of hits and misses — the refusal path skips generation entirely, so an
# all-hits sample would overstate typical latency.
QUERIES = [
    "What are Bullet Kin?",
    "How do I build a GPT from scratch?",
    "What is this document collection about?",
    "asdkjh qwe zxc nonsense gibberish",
]

DEFAULT_REPEATS = 3


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Well-defined for the small samples used here."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(pct / 100 * len(ordered) + 0.5) - 1))
    return ordered[index]


def measure(model: str, repeats: int = DEFAULT_REPEATS) -> dict[str, list[float]]:
    by_node: dict[str, list[float]] = defaultdict(list)
    for _ in range(repeats):
        for query in QUERIES:
            with tracing.collect() as spans:
                ask(query, model=model)
            for span in spans:
                by_node[span["span"]].append(span["duration_ms"])
    return by_node


def main() -> int:
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REPEATS
    total = repeats * len(QUERIES)

    # No default model -- pick whichever is installed, same convention as
    # src.rag.graph.main.
    installed = list_installed()
    if not installed:
        print("no chat model downloaded -- pull one first (see Settings, or "
              "`ollama pull <model>`)")
        return 1
    model = installed[0]

    print(f"model: {model}")
    print(f"running {total} queries ({len(QUERIES)} queries x {repeats} repeats)\n")

    by_node = measure(model, repeats)

    header = f"{'node':<10}{'n':>4}{'mean':>10}{'p50':>10}{'p95':>10}{'p99':>10}{'max':>10}"
    print(header)
    print("-" * len(header))
    # "ask" last: it is the end-to-end total, not a peer of the node spans.
    for node in ("retrieve", "grade", "generate", "ask"):
        samples = by_node.get(node)
        if not samples:
            continue
        print(
            f"{node:<10}{len(samples):>4}"
            f"{statistics.mean(samples):>9.1f}ms"
            f"{_percentile(samples, 50):>9.1f}ms"
            f"{_percentile(samples, 95):>9.1f}ms"
            f"{_percentile(samples, 99):>9.1f}ms"
            f"{max(samples):>9.1f}ms"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
