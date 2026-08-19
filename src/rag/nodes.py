"""Phase 5: LangGraph nodes — retrieve, grade, generate."""

import os
import threading

from langchain_ollama import ChatOllama

from src.observability import tracing
from src.rag.citations import citations_enabled, collect_sources
from src.rag.prompts import RAG_PROMPT, format_context
from src.vectorstore.store import load_vectorstore

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = "llama3.2:3b"
RETRIEVE_K = 5

# Absolute floor: a chunk this dissimilar is noise no matter what else came
# back — this is what lets an off-topic query refuse instead of citing the
# five least-bad chunks in the collection.
RELEVANCE_FLOOR = float(os.environ.get("RAG_RELEVANCE_FLOOR", "0.48"))
# Relative cutoff: drop chunks far weaker than the best hit, so a query with
# one strong match doesn't drag along filler that happens to clear the floor.
RELEVANCE_RATIO = float(os.environ.get("RAG_RELEVANCE_RATIO", "0.9"))

SCORE_KEY = "relevance_score"

_vectorstore = None
_llm = None
# The benchmark runner invokes the graph from a thread pool, so the lazy
# singletons below need guarding -- an unlocked race would build a second
# PGVector engine (and its connection pool) and leak it.
_init_lock = threading.Lock()


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        with _init_lock:
            if _vectorstore is None:
                _vectorstore = load_vectorstore()
    return _vectorstore


def _get_llm():
    global _llm
    if _llm is None:
        with _init_lock:
            if _llm is None:
                _llm = ChatOllama(model=MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    return _llm


@tracing.traced("retrieve")
def retrieve_node(state: dict) -> dict:
    scored = _get_vectorstore().similarity_search_with_relevance_scores(
        state["query"], k=RETRIEVE_K
    )
    docs = []
    for doc, score in scored:
        doc.metadata[SCORE_KEY] = score
        docs.append(doc)
    tracing.detail(
        k=RETRIEVE_K,
        scores=[round(s, 4) for _, s in scored],
        sources=[d.metadata.get("source") for d in docs],
    )
    return {"retrieved_docs": docs}


@tracing.traced("grade")
def grade_node(state: dict) -> dict:
    docs = [d for d in state["retrieved_docs"] if d.page_content.strip()]
    scores = [d.metadata.get(SCORE_KEY, 0.0) for d in docs]
    if not scores:
        tracing.detail(kept=0, dropped=len(state["retrieved_docs"]), cutoff=None)
        return {"graded_docs": []}

    cutoff = max(RELEVANCE_FLOOR, max(scores) * RELEVANCE_RATIO)
    graded = [d for d in docs if d.metadata.get(SCORE_KEY, 0.0) >= cutoff]
    tracing.detail(
        cutoff=round(cutoff, 4),
        # Which bound actually decided the cutoff — the absolute floor or the
        # ratio against the best hit. The distinction explains most refusals.
        bound="floor" if RELEVANCE_FLOOR >= max(scores) * RELEVANCE_RATIO else "ratio",
        kept=len(graded),
        dropped=len(state["retrieved_docs"]) - len(graded),
    )
    return {"graded_docs": graded}


@tracing.traced("generate")
def generate_node(state: dict) -> dict:
    if not state["graded_docs"]:
        tracing.detail(refused=True)
        return {
            "answer": "I don't have enough context to answer that question.",
            "sources": [],
        }

    context = format_context(state["graded_docs"])
    prompt = RAG_PROMPT.format(context=context, question=state["query"])
    response = _get_llm().invoke(prompt)
    sources = collect_sources(state["graded_docs"]) if citations_enabled() else []
    tracing.detail(
        refused=False,
        model=MODEL,
        prompt_chars=len(prompt),
        docs=len(state["graded_docs"]),
        **_token_usage(response),
    )
    return {"answer": response.content, "sources": sources}


def _token_usage(response) -> dict:
    """Pull Ollama's own token counts off the response, if present.

    Ollama reports eval_count/eval_duration per request, which gives true
    tokens-per-second without estimating. Absent on a fake LLM in tests.
    """
    meta = getattr(response, "response_metadata", None) or {}
    usage = {
        k: meta[k]
        for k in ("prompt_eval_count", "eval_count", "eval_duration")
        if k in meta
    }
    if usage.get("eval_count") and usage.get("eval_duration"):
        # eval_duration is nanoseconds.
        usage["tokens_per_sec"] = round(
            usage["eval_count"] / (usage["eval_duration"] / 1e9), 1
        )
    return usage
