"""Phase 5: LangGraph nodes — retrieve, grade, generate."""

import os
import threading

from langchain_ollama import ChatOllama

from src.config import env_flag
from src.observability import tracing
from src.rag.citations import citations_enabled, collect_sources
from src.rag.prompts import RAG_PROMPT, format_context
from src.vectorstore import hybrid
from src.vectorstore.store import load_vectorstore

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = "llama3.2:3b"

# Both served locally by the same Ollama instance -- picking a second model
# means picking a second local weights file, not adding a cloud provider.
# qwen3:4b (Q4, ~2.5GB) is the Tier A pick for this card: it leaves headroom
# under the 3.0GB budget for a reranker to run alongside it, unlike
# qwen2.5:7b's ~4.7GB. Keep this in sync with what's actually pulled
# (`docker exec <ollama container> ollama list`).
AVAILABLE_MODELS = (MODEL, "qwen3:4b")

RETRIEVE_K = 5

# Absolute floor: a chunk this dissimilar is noise no matter what else came
# back — this is what lets an off-topic query refuse instead of citing the
# five least-bad chunks in the collection.
RELEVANCE_FLOOR = float(os.environ.get("RAG_RELEVANCE_FLOOR", "0.56"))
# Relative cutoff: drop chunks far weaker than the best hit, so a query with
# one strong match doesn't drag along filler that happens to clear the floor.
RELEVANCE_RATIO = float(os.environ.get("RAG_RELEVANCE_RATIO", "0.9"))

SCORE_KEY = hybrid.DENSE_SCORE_KEY
LEXICAL_KEY = hybrid.LEXICAL_SCORE_KEY


def hybrid_enabled() -> bool:
    """Fuse full-text results with the dense ones. Set RAG_HYBRID=false to compare."""
    return env_flag("RAG_HYBRID", default=True)

_vectorstore = None
# Keyed by model name rather than a single instance, since a query can now
# pick between AVAILABLE_MODELS -- each gets its own lazily-built client.
_llms: dict[str, ChatOllama] = {}
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


def _get_llm(model: str = MODEL) -> ChatOllama:
    llm = _llms.get(model)
    if llm is None:
        with _init_lock:
            llm = _llms.get(model)
            if llm is None:
                llm = ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=0)
                _llms[model] = llm
    return llm


@tracing.traced("retrieve")
def retrieve_node(state: dict) -> dict:
    store = _get_vectorstore()
    if hybrid_enabled():
        docs = hybrid.retrieve(store, state["query"], k=RETRIEVE_K)
    else:
        docs = []
        for doc, score in store.similarity_search_with_relevance_scores(
            state["query"], k=RETRIEVE_K
        ):
            doc.metadata[SCORE_KEY] = score
            docs.append(doc)

    tracing.detail(
        k=RETRIEVE_K,
        hybrid=hybrid_enabled(),
        scores=[
            None if d.metadata.get(SCORE_KEY) is None else round(d.metadata[SCORE_KEY], 4)
            for d in docs
        ],
        lexical_hits=sum(1 for d in docs if LEXICAL_KEY in d.metadata),
        sources=[d.metadata.get("source") for d in docs],
    )
    return {"retrieved_docs": docs}


@tracing.traced("grade")
def grade_node(state: dict) -> dict:
    docs = [d for d in state["retrieved_docs"] if d.page_content.strip()]
    if not docs:
        tracing.detail(kept=0, dropped=len(state["retrieved_docs"]), cutoff=None)
        return {"graded_docs": []}

    # A full-text hit is kept unconditionally. Postgres only returns rows whose
    # tsvector actually matches the query, so a lexical hit is direct evidence
    # that the chunk contains the query's terms — and an off-topic query gets an
    # empty lexical list rather than a weak one, which is what preserves the
    # refusal path without adding a second threshold to tune.
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

    # Rebuild in fused-rank order rather than concatenating the two groups.
    keep = {id(d) for d in lexical} | {id(d) for d in graded_dense}
    graded = [d for d in docs if id(d) in keep]

    tracing.detail(
        cutoff=None if cutoff is None else round(cutoff, 4),
        # Which bound decided the dense cutoff — the absolute floor or the ratio
        # against the best hit. The distinction explains most refusals.
        bound=None
        if not dense_scores
        else (
            "floor"
            if RELEVANCE_FLOOR >= max(dense_scores) * RELEVANCE_RATIO
            else "ratio"
        ),
        kept_lexical=len(lexical),
        kept_dense=len(graded_dense),
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

    model = state.get("model") or MODEL
    context = format_context(state["graded_docs"])
    prompt = RAG_PROMPT.format(context=context, question=state["query"])
    response = _get_llm(model).invoke(prompt)
    sources = collect_sources(state["graded_docs"]) if citations_enabled() else []
    tracing.detail(
        refused=False,
        model=model,
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
