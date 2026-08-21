"""Prompt template and context formatting for grounded generation."""

RAG_PROMPT = """Answer the question using ONLY the context below.
If the context doesn't contain the answer, say so — don't guess.
Answer in plain prose -- don't cite or name sources inline, citations are \
shown separately.

Context:
{context}

Question: {question}

Answer:"""


def format_context(docs: list) -> str:
    """Join the graded chunks into the prompt's context block.

    Chunks are separated but not labelled. A provenance header used to precede
    each one, which cost prompt budget on every query and invited the model to
    quote a filename back at the reader despite the instruction above -- the
    citations are assembled from metadata and shown beside the answer instead.
    """
    return "\n\n".join(doc.page_content for doc in docs)
