from src.config import get_settings
from src.observability import tracing
from src.vectorstore import hybrid

_settings = get_settings()

RELEVANCE_FLOOR = _settings.relevance_floor
RELEVANCE_RATIO = _settings.relevance_ratio

SCORE_KEY = hybrid.DENSE_SCORE_KEY
LEXICAL_KEY = hybrid.LEXICAL_SCORE_KEY


@tracing.traced("grade")
def grade_node(state: dict) -> dict:
    docs = [d for d in state["retrieved_docs"] if d.page_content.strip()]
    if not docs:
        tracing.detail(kept=0, dropped=len(state["retrieved_docs"]), cutoff=None)
        return {"graded_docs": []}

    lexical = [d for d in docs if LEXICAL_KEY in d.metadata]
    dense_only = [d for d in docs if LEXICAL_KEY not in d.metadata]
    dense_scores = [
        d.metadata[SCORE_KEY]
        for d in dense_only
        if d.metadata.get(SCORE_KEY) is not None
    ]

    cutoff = None
    graded_dense = []
    if dense_scores:
        cutoff = max(RELEVANCE_FLOOR, max(dense_scores) * RELEVANCE_RATIO)
        graded_dense = [
            d for d in dense_only if (d.metadata.get(SCORE_KEY) or 0.0) >= cutoff
        ]

    keep = {id(d) for d in lexical} | {id(d) for d in graded_dense}
    graded = [d for d in docs if id(d) in keep]

    bound = None if not dense_scores else ("floor" if RELEVANCE_FLOOR >= max(dense_scores) * RELEVANCE_RATIO else "ratio")

    tracing.detail(
        cutoff=None if cutoff is None else round(cutoff, 4),
        bound=bound,
        kept_lexical=len(lexical),
        kept_dense=len(graded_dense),
        kept=len(graded),
        dropped=len(state["retrieved_docs"]) - len(graded),
    )
    return {"graded_docs": graded}
