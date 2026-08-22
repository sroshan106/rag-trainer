RAG_PROMPT = """Answer the question using ONLY the context below.
If the context doesn't contain the answer, say so — don't guess.
Answer in plain prose -- don't cite or name sources inline, citations are \
shown separately.

Context:
{context}

Question: {question}

Answer:"""


def format_context(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)
