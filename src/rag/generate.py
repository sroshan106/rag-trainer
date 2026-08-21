"""Generation node: grounded prompt synthesis, token streaming, confidence scoring, and direct answers."""

import time

from src.observability import tracing
from src.rag.citations import citations_enabled, collect_citations
from src.rag.models import get_llm
from src.rag.prompts import RAG_PROMPT, format_context
from src.rag.thinking import NO_THINK_SUFFIX, ThinkFilter, strip_thinking, wants_no_think
from src.vectorstore import hybrid, rerank

REFUSAL_ANSWER = "I don't have enough context to answer that question."
SCORE_KEY = hybrid.DENSE_SCORE_KEY


def confidence_of(docs: list) -> float:
    """How well the best surviving chunk matches the query, on 0-1 scale."""
    scores = [
        score
        for doc in docs
        for score in (
            doc.metadata.get(rerank.RERANK_SCORE_KEY) or doc.metadata.get(SCORE_KEY),
        )
        if score is not None
    ]
    return round(max(scores), 4) if scores else 0.0


def _token_usage(response) -> dict:
    """Extract Ollama token counts from response metadata."""
    meta = getattr(response, "response_metadata", None) or {}
    usage = {
        k: meta[k]
        for k in ("prompt_eval_count", "eval_count", "eval_duration")
        if k in meta
    }
    if usage.get("eval_count") and usage.get("eval_duration"):
        usage["tokens_per_sec"] = round(
            usage["eval_count"] / (usage["eval_duration"] / 1e9), 1
        )
    return usage


def _refusal() -> dict:
    tracing.detail(refused=True, confidence=0.0)
    return {"answer": REFUSAL_ANSWER, "citations": [], "refused": True, "confidence": 0.0}


def prompt_for(state: dict) -> tuple[str, str]:
    """Build the generation prompt. Returns ``(model, prompt)``."""
    from src.rag import nodes

    model = state["model"]
    context = format_context(state["graded_docs"])
    prompt = RAG_PROMPT.format(context=context, question=state["query"])
    wants_check = getattr(nodes, "_wants_no_think", wants_no_think)
    suffix = getattr(nodes, "_NO_THINK_SUFFIX", NO_THINK_SUFFIX)
    if wants_check(model):
        prompt += suffix
    return model, prompt


@tracing.traced("generate")
def generate_node(state: dict) -> dict:
    from src.rag import nodes

    if not state["graded_docs"]:
        refusal_fn = getattr(nodes, "_refusal", _refusal)
        return refusal_fn()

    prompt_fn = getattr(nodes, "_prompt_for", prompt_for)
    get_llm_fn = getattr(nodes, "_get_llm", get_llm)
    strip_fn = getattr(nodes, "_strip_thinking", strip_thinking)
    confidence_fn = getattr(nodes, "confidence_of", confidence_of)

    model, prompt = prompt_fn(state)
    response = get_llm_fn(model).invoke(prompt)
    answer = strip_fn(response.content)
    citations = collect_citations(state["graded_docs"]) if citations_enabled() else []
    confidence = confidence_fn(state["graded_docs"])
    tracing.detail(
        refused=False,
        confidence=confidence,
        model=model,
        prompt_chars=len(prompt),
        docs=len(state["graded_docs"]),
        **_token_usage(response),
    )
    return {
        "answer": answer,
        "citations": citations,
        "refused": False,
        "confidence": confidence,
    }


def direct_answer(model: str, query: str) -> dict:
    """Answer query without retrieval step for side-by-side comparison."""
    from src.rag import nodes

    wants_check = getattr(nodes, "_wants_no_think", wants_no_think)
    suffix = getattr(nodes, "_NO_THINK_SUFFIX", NO_THINK_SUFFIX)
    get_llm_fn = getattr(nodes, "_get_llm", get_llm)
    strip_fn = getattr(nodes, "_strip_thinking", strip_thinking)

    prompt = query + suffix if wants_check(model) else query
    started = time.perf_counter()
    response = get_llm_fn(model).invoke(prompt)
    generate_ms = round((time.perf_counter() - started) * 1000, 1)
    return {
        "answer": strip_fn(response.content),
        "generate_ms": generate_ms,
        **_token_usage(response),
    }


def generate_stream(state: dict):
    """Yield answer text as it arrives; return generate_node's dict at the end."""
    from src.rag import nodes

    refusal_fn = getattr(nodes, "_refusal", _refusal)
    prompt_fn = getattr(nodes, "_prompt_for", prompt_for)
    get_llm_fn = getattr(nodes, "_get_llm", get_llm)
    think_filter_cls = getattr(nodes, "_ThinkFilter", ThinkFilter)
    confidence_fn = getattr(nodes, "confidence_of", confidence_of)

    with tracing.span("generate"):
        if not state["graded_docs"]:
            refusal = refusal_fn()
            yield refusal["answer"]
            return refusal

        model, prompt = prompt_fn(state)
        think = think_filter_cls()
        parts = []
        last = None
        for chunk in get_llm_fn(model).stream(prompt):
            last = chunk
            visible = think.feed(chunk.content)
            if not parts:
                visible = visible.lstrip()
            if visible:
                parts.append(visible)
                yield visible

        answer = "".join(parts).strip()
        citations = collect_citations(state["graded_docs"]) if citations_enabled() else []
        confidence = confidence_fn(state["graded_docs"])
        tracing.detail(
            refused=False,
            confidence=confidence,
            model=model,
            prompt_chars=len(prompt),
            docs=len(state["graded_docs"]),
            streamed=True,
            **_token_usage(last),
        )
        return {
            "answer": answer,
            "citations": citations,
            "refused": False,
            "confidence": confidence,
        }
