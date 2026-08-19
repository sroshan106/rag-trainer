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

from src.config import env_flag

# English-only and tiny, matching the corpus. Trained on MS MARCO passage
# ranking -- short query against a paragraph of prose -- which is the shape of
# every query this pipeline serves.
RERANK_MODEL = os.environ.get(
    "RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

RERANK_SCORE_KEY = "rerank_score"

# CPU keeps the card free; see the module docstring for the latency this costs.
DEVICE = os.environ.get("RAG_RERANK_DEVICE", "cpu")

# Truncation bound for the pair encoder, and the dominant cost knob -- latency
# is linear in it. Chunks run ~1000 tokens against a 512-token budget shared by
# query and document, so the tail of a chunk already goes unscored at the
# default. Reranking is a ranking decision rather than a reading one, and a
# chunk's opening carries its topic; this is also the strongest argument for
# splitting retrieval onto smaller chunks than generation reads.
MAX_LENGTH = int(os.environ.get("RAG_RERANK_MAX_LENGTH", "512"))

_model = None
# The benchmark runner drives the graph from a thread pool, so the lazy
# singleton needs the same guard the vectorstore and LLM clients use -- an
# unlocked race would load a second copy of the weights into RAM.
_init_lock = threading.Lock()


def rerank_enabled() -> bool:
    """Reorder candidates with the cross-encoder. Set RAG_RERANK=false to compare."""
    return env_flag("RAG_RERANK", default=True)


def _get_model():
    """Load the cross-encoder once, pinned to CPU.

    Imported lazily so that importing this module -- which the graph does on
    every run -- costs nothing when reranking is switched off, and so the
    torch dependency is only required by the path that actually uses it.
    """
    global _model
    if _model is None:
        with _init_lock:
            if _model is None:
                import torch
                from sentence_transformers import CrossEncoder

                if DEVICE == "cpu":
                    # Torch defaults to half the visible cores. Nothing else
                    # runs on CPU during a query -- generation is on the card --
                    # so the reranker may as well have all of them.
                    torch.set_num_threads(os.cpu_count() or 1)

                _model = CrossEncoder(
                    RERANK_MODEL, max_length=MAX_LENGTH, device=DEVICE
                )
    return _model


def _squash(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-logit))


def rerank(query: str, docs: list[Document], k: int) -> list[Document]:
    """Return the k documents the cross-encoder ranks highest for ``query``.

    Every input document is stamped with ``rerank_score`` before truncation, so
    a trace shows what the discarded candidates scored and not just which ones
    survived.
    """
    if not docs:
        return []

    scores = _get_model().predict([(query, doc.page_content) for doc in docs])
    for doc, score in zip(docs, scores):
        doc.metadata[RERANK_SCORE_KEY] = _squash(float(score))

    ranked = sorted(docs, key=lambda d: d.metadata[RERANK_SCORE_KEY], reverse=True)
    return ranked[:k]
