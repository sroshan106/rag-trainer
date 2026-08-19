"""Phase 6: prompt template for grounded generation."""

RAG_PROMPT = """Answer the question using ONLY the context below.
If the context doesn't contain the answer, say so — don't guess.
After every sentence that uses a fact from the context, add its source tag in \
parentheses, e.g. "Paris is the capital of France (source: geo.txt)."

Context:
{context}

Question: {question}

Answer:"""


def format_context(docs: list) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[source: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)
