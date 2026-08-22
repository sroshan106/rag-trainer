import time

from src.observability import tracing
from src.rag.citations import citations_enabled, collect_citations
from src.rag.models import get_llm
from src.rag.prompts import RAG_PROMPT, format_context
from src.vectorstore import hybrid, rerank

REFUSAL_ANSWER = "I don't have enough context to answer that question."
SCORE_KEY = hybrid.DENSE_SCORE_KEY

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def strip_thinking(text: str) -> str:
    if THINK_CLOSE in text:
        return text.split(THINK_CLOSE, 1)[1].strip()
    return text.strip()


class ThinkFilter:
    def __init__(self) -> None:
        self._buffer = ""
        self._passthrough = False

    def feed(self, chunk: str) -> str:
        if self._passthrough:
            return chunk
        self._buffer += chunk
        text = self._buffer.lstrip()
        if THINK_CLOSE in text:
            self._passthrough = True
            return text.split(THINK_CLOSE, 1)[1].lstrip()
        if text.startswith(THINK_OPEN) or THINK_OPEN.startswith(text):
            return ""
        self._passthrough = True
        return text


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


def generate_node(state: dict) -> dict:
    generation = generate_stream(state)
    try:
        while True:
            next(generation)
    except StopIteration as stop:
        return stop.value


def direct_answer(model: str, query: str) -> dict:
    started = time.perf_counter()
    response = get_llm(model).invoke(query)
    generate_ms = round((time.perf_counter() - started) * 1000, 1)
    return {
        "answer": strip_thinking(response.content),
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
        think = ThinkFilter()
        parts = []
        last = None
        for chunk in get_llm(model).stream(prompt):
            last = chunk
            visible = think.feed(chunk.content)
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
