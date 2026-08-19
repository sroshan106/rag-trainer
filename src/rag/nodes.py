"""Phase 5: LangGraph nodes — retrieve, grade, generate."""

import os

from langchain_ollama import ChatOllama

from src.rag.citations import citations_enabled, collect_sources
from src.rag.prompts import RAG_PROMPT, format_context
from src.vectorstore.store import load_vectorstore

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
RETRIEVE_K = 5

# Absolute floor: a chunk this dissimilar is noise no matter what else came
# back — this is what lets an off-topic query refuse instead of citing the
# five least-bad chunks in the collection.
RELEVANCE_FLOOR = float(os.environ.get("RAG_RELEVANCE_FLOOR", "0.6"))
# Relative cutoff: drop chunks far weaker than the best hit, so a query with
# one strong match doesn't drag along filler that happens to clear the floor.
RELEVANCE_RATIO = float(os.environ.get("RAG_RELEVANCE_RATIO", "0.8"))

SCORE_KEY = "relevance_score"

_vectorstore = None
_llm = None


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_vectorstore()
    return _vectorstore


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOllama(model="llama3.2:3b", base_url=OLLAMA_BASE_URL, temperature=0)
    return _llm


def retrieve_node(state: dict) -> dict:
    scored = _get_vectorstore().similarity_search_with_relevance_scores(
        state["query"], k=RETRIEVE_K
    )
    docs = []
    for doc, score in scored:
        doc.metadata[SCORE_KEY] = score
        docs.append(doc)
    return {"retrieved_docs": docs}


def grade_node(state: dict) -> dict:
    docs = [d for d in state["retrieved_docs"] if d.page_content.strip()]
    scores = [d.metadata.get(SCORE_KEY, 0.0) for d in docs]
    if not scores:
        return {"graded_docs": []}

    cutoff = max(RELEVANCE_FLOOR, max(scores) * RELEVANCE_RATIO)
    graded = [d for d in docs if d.metadata.get(SCORE_KEY, 0.0) >= cutoff]
    return {"graded_docs": graded}


def generate_node(state: dict) -> dict:
    if not state["graded_docs"]:
        return {
            "answer": "I don't have enough context to answer that question.",
            "sources": [],
        }

    context = format_context(state["graded_docs"])
    prompt = RAG_PROMPT.format(context=context, question=state["query"])
    response = _get_llm().invoke(prompt)
    sources = collect_sources(state["graded_docs"]) if citations_enabled() else []
    return {"answer": response.content, "sources": sources}
