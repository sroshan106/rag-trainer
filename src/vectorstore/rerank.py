"""Cross-encoder reranking over the fused candidate list.

Dense and lexical retrieval both score the query against a document
*independently* -- neither ever sees the pair together, so a chunk that merely
shares vocabulary with the query can outrank the one that answers it. A
cross-encoder reads query and chunk in a single forward pass and scores the
pair directly. That is what separates near-duplicate passages describing
different entities, which is the failure mode a wiki-shaped corpus is full of.

This runs on CPU by default. The GPU is the contended resource -- the
generation model and the embedding model already share 4GB -- while the CPU
sits idle for the whole of generation, so the card keeps its entire budget for
the models that have no CPU option.

Measured cost of that choice on this box (8 cores, 20 candidates of full-length
chunks, all of which saturate the 512-token pair budget):

    max_length=512, k=20   2400ms
    max_length=256, k=20   1125ms
    max_length=512, k=10   1200ms
    max_length=128, k=20    750ms

Seconds, not milliseconds -- a cross-encoder is a full transformer forward pass
per pair, and 20 pairs at 512 tokens is real work. It is affordable here only
because generation on this hardware costs 9s and up, so reranking is a fraction
of a query rather than a doubling of it. Tune ``RAG_RERANK_MAX_LENGTH`` down
before ``RAG_FETCH_K``: candidate depth is what the reranker is for, while the
tail of a 1000-token chunk contributes little to a ranking decision.

``RAG_RERANK_DEVICE=cuda`` moves it to the card, which this model is small
enough to make defensible -- 22M parameters is roughly 90MB of weights, not the
hundreds of megabytes a full-size reranker would claim. Worth measuring against
the VRAM it takes from the generation model before adopting.

Scores are squashed through a logistic to land on 0-1. Cross-encoders emit raw
logits on an arbitrary scale; the squash is monotonic, so it changes no
ordering, but it puts the score on the same footing as the cosine and fusion
scores already in the metadata and gives a later cutoff something calibrated to
threshold against.
"""

import math
import os
import threading

from langchain_core.documents import Document

from src.config import env_flag, get_settings
from src.observability.logging import log

_settings = get_settings()

# English-only and tiny, matching the corpus. Trained on MS MARCO passage
# ranking -- short query against a paragraph of prose -- which is the shape of
# every query this pipeline serves.
RERANK_MODEL = _settings.rerank_model

RERANK_SCORE_KEY = "rerank_score"

DEVICE = _settings.rerank_device
MAX_LENGTH = _settings.rerank_max_length

_model = None
# The benchmark runner drives the graph from a thread pool, so the lazy
# singleton needs the same guard the vectorstore and LLM clients use -- an
# unlocked race would load a second copy of the weights into RAM.
_init_lock = threading.Lock()


def rerank_enabled() -> bool:
    """Reorder candidates with the cross-encoder. Set RAG_RERANK=false to compare."""
    return env_flag("RAG_RERANK", default=True)


def hf_cached_repos() -> set[str]:
    """Repo ids present in the local HuggingFace cache. Empty when unavailable."""
    try:
        from huggingface_hub import CacheNotFound, scan_cache_dir
    except ImportError:
        return set()
    try:
        return {repo.repo_id for repo in scan_cache_dir().repos}
    except CacheNotFound:
        return set()
    except Exception as exc:  # noqa: BLE001 - a status check must not 500 the page
        log("warning", "huggingface cache probe failed", error=repr(exc))
        return set()


def _get_model():
    """Load the cross-encoder lazily, pinned to device."""
    global _model
    if _model is None:
        with _init_lock:
            if _model is None:
                import torch
                from sentence_transformers import CrossEncoder

                if DEVICE == "cpu":
                    torch.set_num_threads(os.cpu_count() or 1)

                _model = CrossEncoder(
                    RERANK_MODEL, max_length=MAX_LENGTH, device=DEVICE
                )
    return _model


def ensure_loaded(model: str | None = None) -> None:
    """Force the cross-encoder to load immediately."""
    target = model or RERANK_MODEL
    if target == RERANK_MODEL:
        _get_model()
    else:
        from sentence_transformers import CrossEncoder

        CrossEncoder(target, max_length=MAX_LENGTH, device=DEVICE)


def _squash(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-logit))


def rerank(query: str, docs: list[Document], k: int) -> list[Document]:
    """Return the top k documents ranked by the cross-encoder."""
    if not docs:
        return []

    if RERANK_MODEL not in hf_cached_repos():
        # Reranker isn't downloaded -- don't silently pull it mid-query.
        # Download it explicitly from Settings > Rerankers instead.
        log("warning", "reranker not installed, skipping rerank", model=RERANK_MODEL)
        return docs[:k]

    scores = _get_model().predict([(query, doc.page_content) for doc in docs])
    for doc, score in zip(docs, scores):
        doc.metadata[RERANK_SCORE_KEY] = _squash(float(score))

    ranked = sorted(docs, key=lambda d: d.metadata[RERANK_SCORE_KEY], reverse=True)
    return ranked[:k]
