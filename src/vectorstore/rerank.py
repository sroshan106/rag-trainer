import math
import os
import threading

from langchain_core.documents import Document

from src.config import env_flag, get_settings
from src.observability.logging import log

_settings = get_settings()

RERANK_MODEL = _settings.rerank_model

RERANK_SCORE_KEY = "rerank_score"

DEVICE = _settings.rerank_device
MAX_LENGTH = _settings.rerank_max_length

_model = None
_init_lock = threading.Lock()

_resolved_device = None
_device_lock = threading.Lock()


def _resolve_device() -> str:
    global _resolved_device
    if _resolved_device is None:
        with _device_lock:
            if _resolved_device is None:
                if DEVICE == "cuda":
                    import torch

                    if torch.cuda.is_available():
                        _resolved_device = "cuda"
                    else:
                        log("warning", "RAG_RERANK_DEVICE=cuda requested but no GPU found, using cpu")
                        _resolved_device = "cpu"
                else:
                    _resolved_device = DEVICE
    return _resolved_device


def rerank_enabled() -> bool:
    return env_flag("RAG_RERANK", default=True)


def hf_cached_repos() -> set[str]:
    try:
        from huggingface_hub import CacheNotFound, scan_cache_dir
    except ImportError:
        return set()
    try:
        return {repo.repo_id for repo in scan_cache_dir().repos}
    except CacheNotFound:
        return set()
    except Exception as exc:
        log("warning", "huggingface cache probe failed", error=repr(exc))
        return set()


def _get_model():
    global _model
    if _model is None:
        with _init_lock:
            if _model is None:
                import torch
                from sentence_transformers import CrossEncoder

                device = _resolve_device()
                if device == "cpu":
                    torch.set_num_threads(os.cpu_count() or 1)

                _model = CrossEncoder(
                    RERANK_MODEL, max_length=MAX_LENGTH, device=device
                )
    return _model


def ensure_loaded(model: str | None = None) -> None:
    target = model or RERANK_MODEL
    if target == RERANK_MODEL:
        _get_model()
    else:
        from sentence_transformers import CrossEncoder

        CrossEncoder(target, max_length=MAX_LENGTH, device=_resolve_device())


def _squash(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-logit))


def rerank(query: str, docs: list[Document], k: int) -> list[Document]:
    if not docs:
        return []

    if RERANK_MODEL not in hf_cached_repos():
        log("warning", "reranker not installed, skipping rerank", model=RERANK_MODEL)
        return docs[:k]

    scores = _get_model().predict([(query, doc.page_content) for doc in docs])
    for doc, score in zip(docs, scores):
        doc.metadata[RERANK_SCORE_KEY] = _squash(float(score))

    ranked = sorted(docs, key=lambda d: d.metadata[RERANK_SCORE_KEY], reverse=True)
    return ranked[:k]
