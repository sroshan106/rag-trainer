import time

from src.observability import tracing
from src.rag.citations import citations_enabled, collect_citations
from src.rag.models import get_llm
from src.rag.prompts import RAG_PROMPT, format_context
from src.vectorstore import hybrid, rerank

REFUSAL_ANSWER = "I don't have enough context to answer that question."
SCORE_KEY = hybrid.DENSE_SCORE_KEY


def confidence_of(docs: list) -> float:
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
    model = state["model"]
    context = format_context(state["graded_docs"])
    prompt = RAG_PROMPT.format(context=context, question=state["query"])
    return model, prompt


@tracing.traced("generate")
def generate_node(state: dict) -> dict:
    if not state["graded_docs"]:
        return _refusal()

    model, prompt = prompt_for(state)
    response = get_llm(model).invoke(prompt)
    answer = response.content.strip()
    citations = collect_citations(state["graded_docs"]) if citations_enabled() else []
    confidence = confidence_of(state["graded_docs"])
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
    started = time.perf_counter()
    response = get_llm(model).invoke(query)
    generate_ms = round((time.perf_counter() - started) * 1000, 1)
    return {
        "answer": response.content.strip(),
        "generate_ms": generate_ms,
        **_token_usage(response),
    }


def generate_stream(state: dict):
    with tracing.span("generate"):
        if not state["graded_docs"]:
            refusal = _refusal()
            yield refusal["answer"]
            return refusal

        model, prompt = prompt_for(state)
        parts = []
        last = None
        for chunk in get_llm(model).stream(prompt):
            last = chunk
            visible = chunk.content
            if not parts:
                visible = visible.lstrip()
            if visible:
                parts.append(visible)
                yield visible

        answer = "".join(parts).strip()
        citations = collect_citations(state["graded_docs"]) if citations_enabled() else []
        confidence = confidence_of(state["graded_docs"])
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
