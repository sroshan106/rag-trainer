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

# Every chat model this card can run, sized to fit 4GB VRAM alongside the
# reranker (~3.0GB budget -- see rerank.py). No entry here is "the default":
# a query must name one explicitly, and the API layer only accepts a name
# that's both in this tuple and actually pulled (src.rag.model_catalog) --
# there is deliberately no fallback that lets an un-downloaded model answer.
# Keep in sync with what's actually offered for download
# (src.rag.model_catalog.CATALOG mirrors this).
AVAILABLE_MODELS = (
    "llama3.2:3b",
    "llama3.2:1b",
    "qwen3:4b",
    "qwen2.5:3b",
    "gemma2:2b",
    "phi3.5",
)

RETRIEVE_K = 5

# How many candidates retrieval hands the reranker. The cross-encoder can only
# reorder what it is given, so the top-k the grader sees is capped by what came
# back here -- a chunk that lands at rank 7 in the fused list is unrecoverable
# at RETRIEVE_K=5 no matter how well it would have scored. Widening the fetch
# is what makes reranking able to pay for itself; it costs one larger SQL
# result and 20 cross-encoder pairs, not 20 LLM calls.
FETCH_K = int(os.environ.get("RAG_FETCH_K", "20"))

# Ollama defaults num_ctx to 4096. Benchmark prompts on this corpus measure
# ~3.5k tokens, which leaves no margin -- and an overflow is silent, truncating
# from the front of the prompt, which is exactly where the highest-ranked
# context sits. Set explicitly so the failure would be visible instead.
NUM_CTX = int(os.environ.get("RAG_NUM_CTX", "8192"))

# qwen3:4b does not fit this 4GB card at NUM_CTX=8192 -- Ollama spills part of
# it to CPU, and CPU prefill on a multi-thousand-token RAG prompt measured
# 400s+ for a single query. Measured empirically (`ollama ps` CPU/GPU split
# while stepping num_ctx down): 3072 is the largest context that still loads
# 100% GPU (3.0GB) on this card; 4096 already spills 6% to CPU. This trades
# away some of the truncation margin NUM_CTX was set to buy back -- a long
# RAG prompt can still overflow 3072 and lose its earliest (highest-ranked)
# context -- but a slow-and-correct prompt is moot if the query times out
# first.
QWEN3_NUM_CTX = int(os.environ.get("RAG_NUM_CTX_QWEN3", "3072"))


def _num_ctx_for(model: str) -> int:
    return QWEN3_NUM_CTX if model.startswith("qwen3") else NUM_CTX

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


# qwen3's Ollama template always primes the assistant turn with a literal
# `<think>`, regardless of the `reasoning`/`think` request flag -- this
# server's template has no branch that skips it (see docker-compose.yml's
# ollama image pin). "/no_think" is Qwen3's own soft switch: appended to the
# user turn, it makes the model close the thinking block immediately instead
# of spending tokens on it -- a real latency win, not just a display fix.
# Model-specific rather than a generic ChatOllama kwarg because it's a
# convention this model family reads out of the prompt text itself.
_NO_THINK_SUFFIX = " /no_think"
_THINKING_MODEL_PREFIXES = ("qwen3",)


def _wants_no_think(model: str) -> bool:
    return model.startswith(_THINKING_MODEL_PREFIXES)


# Belt-and-suspenders: /no_think usually means the response has no <think>
# block at all, but stripping one if it's there keeps the answer clean even
# if a future model ignores the suffix.
_THINK_CLOSE = "</think>"


def _strip_thinking(text: str) -> str:
    if _THINK_CLOSE in text:
        return text.split(_THINK_CLOSE, 1)[1].strip()
    return text.strip()


REFUSAL_ANSWER = "I don't have enough context to answer that question."


class _ThinkFilter:
    """Drop a leading ``<think>`` block from a stream, token by token.

    ``_strip_thinking`` can only run on a finished answer. Streaming needs the
    same rule applied incrementally: hold text back while it still might be the
    opening of a think block, and release everything after the close tag. A
    model that never opens one pays only the first token of delay.
    """

    _OPEN = "<think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._passthrough = False

    def feed(self, chunk: str) -> str:
        """Return the part of ``chunk`` that is safe to show now."""
        if self._passthrough:
            return chunk
        self._buffer += chunk
        text = self._buffer.lstrip()
        if _THINK_CLOSE in text:
            self._passthrough = True
            return text.split(_THINK_CLOSE, 1)[1].lstrip()
        # Still ambiguous while the text so far is a prefix of "<think>".
        if text.startswith(self._OPEN) or self._OPEN.startswith(text):
            return ""
        self._passthrough = True
        return text


def _refusal() -> dict:
    tracing.detail(refused=True)
    return {"answer": REFUSAL_ANSWER, "sources": []}


def _prompt_for(state: dict) -> tuple[str, str]:
    """Build the generation prompt. Returns ``(model, prompt)``."""
    # No fallback: graph.py rejects an unresolved model before a node ever
    # runs, so state["model"] is always a validated, installed model by now.
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

    Written alongside ``generate_node`` rather than wrapping it: only the
    streaming path can surface a partial answer, and the CLI and benchmark
    callers have no use for a token-by-token path they would only re-join.
    Consume it with ``result = yield from generate_stream(state)``.

    Closing the generator early (the client hung up, or hit Cancel) propagates
    GeneratorExit into ``llm.stream``, which is what actually stops Ollama
    generating instead of letting an abandoned query run to completion.
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
            # Whatever follows a think block starts with the newlines that
            # closed it; leading blank space in a streamed answer is visible
            # in a way it never is in one returned whole.
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
            # Ollama reports its counts on the final chunk of a stream, so the
            # usage fields come from there rather than from a whole response.
            **_token_usage(last),
        )
        return {"answer": answer, "sources": sources}


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
