"""Retrieval node: dense k-NN + lexical search + Reciprocal Rank Fusion + Cross-Encoder reranking."""

import threading

from src.config import env_flag, get_settings
from src.observability import tracing
from src.vectorstore import hybrid, rerank
from src.vectorstore.store import load_vectorstore

_settings = get_settings()

RETRIEVE_K = _settings.retrieve_k
FETCH_K = _settings.fetch_k

SCORE_KEY = hybrid.DENSE_SCORE_KEY
LEXICAL_KEY = hybrid.LEXICAL_SCORE_KEY

_vectorstore = None
_vs_lock = threading.Lock()


def hybrid_enabled() -> bool:
    """Fuse full-text results with the dense ones."""
    return env_flag("RAG_HYBRID", default=True)


def get_vectorstore():
    """Lazily load query-path vectorstore."""
    global _vectorstore
    if _vectorstore is None:
        with _vs_lock:
            if _vectorstore is None:
                _vectorstore = load_vectorstore()
    return _vectorstore


@tracing.traced("retrieve")
def retrieve_node(state: dict) -> dict:
    store = get_vectorstore()
    reranking = rerank.rerank_enabled()
    fetch_k = FETCH_K if reranking else RETRIEVE_K

    if hybrid_enabled():
        docs = hybrid.retrieve(store, state["query"], k=fetch_k)
    else:
        docs = []
        for doc, score in store.similarity_search_with_relevance_scores(
            state["query"], k=fetch_k
        ):
            doc.metadata[SCORE_KEY] = score
            docs.append(doc)

    if reranking:
        fetched = len(docs)
        with tracing.span("rerank"):
            docs = rerank.rerank(state["query"], docs, k=RETRIEVE_K)
            tracing.detail(
                model=rerank.RERANK_MODEL,
                fetched=fetched,
                kept=len(docs),
                scores=[
                    round(d.metadata[rerank.RERANK_SCORE_KEY], 4) for d in docs
                ],
            )

    tracing.detail(
        k=RETRIEVE_K,
        fetch_k=fetch_k,
        reranked=reranking,
        hybrid=hybrid_enabled(),
        scores=[
            None if d.metadata.get(SCORE_KEY) is None else round(d.metadata[SCORE_KEY], 4)
            for d in docs
        ],
        lexical_hits=sum(1 for d in docs if LEXICAL_KEY in d.metadata),
        units=[
            f"{d.metadata.get('filename')}:{d.metadata.get('unit_index')}" for d in docs
        ],
    )
    return {"retrieved_docs": docs}
