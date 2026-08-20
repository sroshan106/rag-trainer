"""Resumable result cache for the benchmark runner.

A benchmark pass is ~120 live LLM calls. Interrupting one and starting over
throws away every completed answer, so results are appended to a JSONL file as
they finish and replayed on the next run.

The cache is keyed by a fingerprint of the configuration that can change an
answer -- model, embedding model, k, the reranker, and the two grading
thresholds. Changing any of them writes to a different file, so a run after a
threshold change can never silently reuse answers produced under the old
cutoff, and a rerank-on/rerank-off comparison cannot answer itself from the
other side's cache.
"""

import hashlib
import json
import threading
from pathlib import Path

from src.rag import nodes
from src.vectorstore import rerank
from src.vectorstore.store import EMBED_MODEL

CACHE_DIR = Path(__file__).parent / ".cache"


def config_fingerprint(model: str) -> str:
    """Short hash of every setting that could change a cached answer.

    Read off the module rather than imported by name: the sweep rebinds the
    cutoffs between grid points, and imported copies would leave every point
    sharing one cache file.

    ``model`` has no fallback -- the caller must already have resolved a real,
    installed model. It must be part of the key, not a detail beside it -- two
    models answer the same question differently, and sharing a cache file
    between them would serve llama's answers as qwen's.
    """
    payload = json.dumps(
        {
            "model": model,
            "embed_model": EMBED_MODEL,
            "k": nodes.RETRIEVE_K,
            "fetch_k": nodes.FETCH_K,
            "rerank": rerank.rerank_enabled(),
            "rerank_model": rerank.RERANK_MODEL,
            "floor": nodes.RELEVANCE_FLOOR,
            "ratio": nodes.RELEVANCE_RATIO,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


class ResultCache:
    """Append-only JSONL store of per-question results, keyed by (suite, question).

    Writes are flushed per record rather than buffered, so a Ctrl-C mid-run
    leaves every completed question on disk.
    """

    def __init__(self, path: Path, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, str], dict] = {}
        self._handle = None

    def __enter__(self):
        if not self.enabled:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()
        self._handle = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        return False

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A run killed mid-write can leave a truncated final line.
                    continue
                self._entries[(record["suite"], record["question"])] = record

    def get(self, suite: str, question: str) -> dict | None:
        if not self.enabled:
            return None
        return self._entries.get((suite, question))

    def put(self, suite: str, question: str, result: dict) -> None:
        if not self.enabled:
            return
        record = {"suite": suite, "question": question, **result}
        with self._lock:
            self._entries[(suite, question)] = record
            self._handle.write(json.dumps(record) + "\n")
            self._handle.flush()

    def __len__(self) -> int:
        return len(self._entries)
