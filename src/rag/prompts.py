"""Phase 6: prompt template for grounded generation."""

from src.rag.citations import UNKNOWN_SOURCE

RAG_PROMPT = """Answer the question using ONLY the context below.
If the context doesn't contain the answer, say so — don't guess.
Answer in plain prose -- don't cite or name sources inline, citations are \
shown separately.

Context:
{context}

Question: {question}

Answer:"""


def format_context(docs: list) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source") or UNKNOWN_SOURCE
        parts.append(f"[source: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)
