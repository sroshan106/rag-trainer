import hashlib
import json
import threading
from pathlib import Path

from src.rag import grade, retrieve
from src.vectorstore import rerank
from src.vectorstore.store import EMBED_MODEL

CACHE_DIR = Path("tests/benchmark/.cache")


def config_fingerprint(model: str) -> str:
    payload = json.dumps(
        {
            "model": model,
            "embed_model": EMBED_MODEL,
            "k": retrieve.RETRIEVE_K,
            "fetch_k": retrieve.FETCH_K,
            "rerank": rerank.rerank_enabled(),
            "rerank_model": rerank.RERANK_MODEL,
            "floor": grade.RELEVANCE_FLOOR,
            "ratio": grade.RELEVANCE_RATIO,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


class ResultCache:

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
