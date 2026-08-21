"""Grading node: relevance filtering, dual cutoffs, and lexical hit preservation."""

import os
from src.observability import tracing
from src.vectorstore import hybrid

RELEVANCE_FLOOR = float(os.environ.get("RAG_RELEVANCE_FLOOR", "0.56"))
RELEVANCE_RATIO = float(os.environ.get("RAG_RELEVANCE_RATIO", "0.9"))

SCORE_KEY = hybrid.DENSE_SCORE_KEY
LEXICAL_KEY = hybrid.LEXICAL_SCORE_KEY


@tracing.traced("grade")
def grade_node(state: dict) -> dict:
    from src.rag import nodes

    score_key = getattr(nodes, "SCORE_KEY", SCORE_KEY)
    lexical_key = getattr(nodes, "LEXICAL_KEY", LEXICAL_KEY)
    relevance_floor = getattr(nodes, "RELEVANCE_FLOOR", RELEVANCE_FLOOR)
    relevance_ratio = getattr(nodes, "RELEVANCE_RATIO", RELEVANCE_RATIO)

    docs = [d for d in state["retrieved_docs"] if d.page_content.strip()]
    if not docs:
        tracing.detail(kept=0, dropped=len(state["retrieved_docs"]), cutoff=None)
        return {"graded_docs": []}

    lexical = [d for d in docs if lexical_key in d.metadata]
    dense_only = [d for d in docs if lexical_key not in d.metadata]
    dense_scores = [
        d.metadata[score_key]
        for d in dense_only
        if d.metadata.get(score_key) is not None
    ]

    cutoff = None
    graded_dense = []
    if dense_scores:
        cutoff = max(relevance_floor, max(dense_scores) * relevance_ratio)
        graded_dense = [
            d for d in dense_only if (d.metadata.get(score_key) or 0.0) >= cutoff
        ]

    keep = {id(d) for d in lexical} | {id(d) for d in graded_dense}
    graded = [d for d in docs if id(d) in keep]

    bound = None if not dense_scores else ("floor" if relevance_floor >= max(dense_scores) * relevance_ratio else "ratio")

    tracing.detail(
        cutoff=None if cutoff is None else round(cutoff, 4),
        bound=bound,
        kept_lexical=len(lexical),
        kept_dense=len(graded_dense),
        kept=len(graded),
        dropped=len(state["retrieved_docs"]) - len(graded),
    )
    return {"graded_docs": graded}
