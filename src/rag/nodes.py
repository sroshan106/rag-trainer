"""Phase 5: LangGraph nodes — retrieve, grade, generate."""

import os
import threading

from langchain_ollama import ChatOllama

from src.config import env_flag
from src.observability import tracing
from src.rag.citations import citations_enabled, collect_sources
from src.rag.prompts import RAG_PROMPT, format_context
from src.vectorstore import hybrid, rerank
from src.vectorstore.store import load_vectorstore

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Chat models sized for 4GB VRAM. Must be in sync with model_catalog.CATALOG.
AVAILABLE_MODELS = (
    "llama3.2:3b",
    "llama3.2:1b",
    "qwen3:4b",
    "qwen2.5:3b",
    "gemma2:2b",
    "phi3.5",
)

RETRIEVE_K = 5

# Retrieval candidate count for reranking; wider fetch lets the reranker find better results.
FETCH_K = int(os.environ.get("RAG_FETCH_K", "20"))

# Explicit context window; Ollama's 4096 default is too small for RAG prompts.
NUM_CTX = int(os.environ.get("RAG_NUM_CTX", "8192"))

# qwen3:4b at 8192 ctx spills to CPU (400s+ per query). 3072 is the largest that fits fully on a 4GB card.
QWEN3_NUM_CTX = int(os.environ.get("RAG_NUM_CTX_QWEN3", "3072"))


def _num_ctx_for(model: str) -> int:
    return QWEN3_NUM_CTX if model.startswith("qwen3") else NUM_CTX

# Floor below which any chunk is noise, enabling off-topic refusal.
RELEVANCE_FLOOR = float(os.environ.get("RAG_RELEVANCE_FLOOR", "0.56"))
# Relative cutoff: drop chunks far weaker than the best hit.
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


def _get_llm(model: str) -> ChatOllama:
    llm = _llms.get(model)
    if llm is None:
        with _init_lock:
            llm = _llms.get(model)
            if llm is None:
                llm = ChatOllama(
                    model=model,
                    base_url=OLLAMA_BASE_URL,
                    temperature=0,
                    num_ctx=_num_ctx_for(model),
                    # Qwen3 thinks by default; a <think> block in the answer
                    # would leak straight into the UI and cost tokens/latency
                    # for no benefit on a single-hop factual RAG answer.
                    # Ignored by models (like llama3.2) that don't support it.
                    reasoning=False,
                )
                _llms[model] = llm
    return llm


@tracing.traced("retrieve")
def retrieve_node(state: dict) -> dict:
    store = _get_vectorstore()
    reranking = rerank.rerank_enabled()
    # Fetch wide only when something downstream will narrow the list back down;
    # without the reranker the extra candidates would go straight to the grader
    # and dilute it.
    fetch_k = FETCH_K if reranking else RETRIEVE_K

    if hybrid_enabled():
        docs = hybrid.retrieve(store, state["query"], k=fetch_k)
    else:
        docs = []
        for doc, score in store.similarity_search_with_relevance_scores(
            state["query"], k=fetch_k
        ):
            doc.metadata[SCORE_KEY] = score
            docs.append(doc)

    if reranking:
        fetched = len(docs)
        with tracing.span("rerank"):
            docs = rerank.rerank(state["query"], docs, k=RETRIEVE_K)
            tracing.detail(
                model=rerank.RERANK_MODEL,
                fetched=fetched,
                kept=len(docs),
                scores=[
                    round(d.metadata[rerank.RERANK_SCORE_KEY], 4) for d in docs
                ],
            )

    tracing.detail(
        k=RETRIEVE_K,
        fetch_k=fetch_k,
        reranked=reranking,
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


# "/no_think" is Qwen3's soft switch to immediately close the think block.
# Model-specific since it is read out of the prompt text itself.
_NO_THINK_SUFFIX = " /no_think"
_THINKING_MODEL_PREFIXES = ("qwen3",)


def _wants_no_think(model: str) -> bool:
    return model.startswith(_THINKING_MODEL_PREFIXES)


# Strip a <think> block if present to keep the answer clean.
_THINK_CLOSE = "</think>"


def _strip_thinking(text: str) -> str:
    if _THINK_CLOSE in text:
        return text.split(_THINK_CLOSE, 1)[1].strip()
    return text.strip()


REFUSAL_ANSWER = "I don't have enough context to answer that question."


class _ThinkFilter:
    """Drop a leading ``<think>`` block from a stream, token by token."""

    _OPEN = "<think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._passthrough = False

    def feed(self, chunk: str) -> str:
        if self._passthrough:
            return chunk
        self._buffer += chunk
        text = self._buffer.lstrip()
        if _THINK_CLOSE in text:
            self._passthrough = True
            return text.split(_THINK_CLOSE, 1)[1].lstrip()
        if text.startswith(self._OPEN) or self._OPEN.startswith(text):
            return ""
        self._passthrough = True
        return text


def _refusal() -> dict:
    tracing.detail(refused=True)
    return {"answer": REFUSAL_ANSWER, "sources": []}


def _prompt_for(state: dict) -> tuple[str, str]:
    """Build the generation prompt. Returns ``(model, prompt)``."""
    model = state["model"]
    context = format_context(state["graded_docs"])
    prompt = RAG_PROMPT.format(context=context, question=state["query"])
    if _wants_no_think(model):
        prompt += _NO_THINK_SUFFIX
    return model, prompt


@tracing.traced("generate")
def generate_node(state: dict) -> dict:
    if not state["graded_docs"]:
        return _refusal()

    model, prompt = _prompt_for(state)
    response = _get_llm(model).invoke(prompt)
    answer = _strip_thinking(response.content)
    sources = collect_sources(state["graded_docs"]) if citations_enabled() else []
    tracing.detail(
        refused=False,
        model=model,
        prompt_chars=len(prompt),
        docs=len(state["graded_docs"]),
        **_token_usage(response),
    )
    return {"answer": answer, "sources": sources}


def generate_stream(state: dict):
    """Yield answer text as it arrives; return generate_node's dict at the end.
    
    Closing early propagates GeneratorExit to stop Ollama.
    """
    with tracing.span("generate"):
        if not state["graded_docs"]:
            refusal = _refusal()
            yield refusal["answer"]
            return refusal

        model, prompt = _prompt_for(state)
        think = _ThinkFilter()
        parts = []
        last = None
        for chunk in _get_llm(model).stream(prompt):
            last = chunk
            visible = think.feed(chunk.content)
            # Strip leading whitespace after a think block so it isn't visible in the stream.
            if not parts:
                visible = visible.lstrip()
            if visible:
                parts.append(visible)
                yield visible

        answer = "".join(parts).strip()
        sources = collect_sources(state["graded_docs"]) if citations_enabled() else []
        tracing.detail(
            refused=False,
            model=model,
            prompt_chars=len(prompt),
            docs=len(state["graded_docs"]),
            streamed=True,
            # Ollama reports counts on the final chunk of a stream.
            **_token_usage(last),
        )
        return {"answer": answer, "sources": sources}


def _token_usage(response) -> dict:
    """Extract Ollama token counts from response metadata."""
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
