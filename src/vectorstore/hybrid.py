from langchain_core.documents import Document

from src.config import get_settings
from src.vectorstore import lexical

RRF_K = get_settings().rrf_k

DENSE_SCORE_KEY = "relevance_score"
LEXICAL_SCORE_KEY = "lexical_score"
FUSION_SCORE_KEY = "fusion_score"


def _fuse(ranked_lists: list[list[Document]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for docs in ranked_lists:
        for rank, doc in enumerate(docs, start=1):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
    return scores


def retrieve(
    vectorstore,
    query: str,
    k: int,
    connection: str | None = None,
) -> list[Document]:
    dense_scored = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    lexical_scored = lexical.search(query, k=k, connection=connection)

    merged: dict[str, Document] = {}
    for doc, score in dense_scored:
        doc.metadata[DENSE_SCORE_KEY] = score
        merged[doc.page_content] = doc
    for doc, score in lexical_scored:
        existing = merged.get(doc.page_content)
        if existing is None:
            merged[doc.page_content] = doc
            existing = doc
        existing.metadata[LEXICAL_SCORE_KEY] = score

    fused = _fuse([[d for d, _ in dense_scored], [d for d, _ in lexical_scored]])
    for text, score in fused.items():
        merged[text].metadata[FUSION_SCORE_KEY] = score

    ranked = sorted(
        merged.values(),
        key=lambda d: (
            d.metadata.get(FUSION_SCORE_KEY, 0.0),
            d.metadata.get(LEXICAL_SCORE_KEY, 0.0),
            d.metadata.get(DENSE_SCORE_KEY, 0.0),
        ),
        reverse=True,
    )
    return ranked[:k]
