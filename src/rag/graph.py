import sys
import time

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from src.observability import tracing
from src.rag import history
from src.rag.citations import format_citations
from src.rag.generate import direct_answer, generate_node, generate_stream
from src.rag.grade import grade_node
from src.rag.model_policy import resolve_model
from src.rag.retrieve import retrieve_node


class RAGState(TypedDict):
    query: str
    model: str
    retrieved_docs: list[Document]
    graded_docs: list[Document]
    answer: str
    citations: list[dict]
    refused: bool
    confidence: float


def build_graph():
    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_edge("grade", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


graph = build_graph()


def _run_graph(query: str, resolved_model: str) -> tuple[dict, dict, float]:
    started = time.perf_counter()
    with tracing.collect() as spans, tracing.span("ask"):
        result = graph.invoke({"query": query, "model": resolved_model})
    durations = {s["span"]: s["duration_ms"] for s in spans}
    return result, durations, round((time.perf_counter() - started) * 1000, 1)


def ask(query: str, model: str | None = None) -> dict:
    resolved_model = resolve_model(model)
    entry_id = history.start(query=query, model=resolved_model)
    try:
        result, durations, latency_ms = _run_graph(query, resolved_model)
    except Exception:
        history.fail(entry_id)
        raise

    answer = result["answer"]
    citations = result.get("citations", [])
    refused = result.get("refused", False)
    confidence = result.get("confidence", 0.0)
    history.complete(
        entry_id,
        answer=answer,
        citations=citations,
        refused=refused,
        confidence=confidence,
        latency_ms=latency_ms,
        rerank_ms=durations.get("rerank"),
        generate_ms=durations.get("generate"),
    )
    return {
        "answer": answer,
        "citations": citations,
        "refused": refused,
        "confidence": confidence,
        "id": entry_id,
    }


def ask_compare(query: str, model: str | None = None) -> dict:
    resolved_model = resolve_model(model)
    result, durations, latency_ms = _run_graph(query, resolved_model)
    return {
        "answer": result["answer"],
        "citations": result.get("citations", []),
        "refused": result.get("refused", False),
        "confidence": result.get("confidence", 0.0),
        "model": resolved_model,
        "latency_ms": latency_ms,
        "rerank_ms": durations.get("rerank"),
        "generate_ms": durations.get("generate"),
    }


def ask_direct(query: str, model: str | None = None) -> dict:
    resolved_model = resolve_model(model)
    started = time.perf_counter()
    result = direct_answer(resolved_model, query)
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    return {
        "answer": result["answer"],
        "model": resolved_model,
        "latency_ms": latency_ms,
        "eval_count": result.get("eval_count"),
        "tokens_per_sec": result.get("tokens_per_sec"),
    }


def ask_stream(query: str, model: str | None = None):
    resolved_model = resolve_model(model)
    started = time.perf_counter()
    entry_id = history.start(query=query, model=resolved_model)
    parts: list[str] = []
    try:
        with tracing.collect() as spans, tracing.span("ask"):
            state: dict = {"query": query, "model": resolved_model}

            yield {"type": "stage", "stage": "retrieve"}
            state.update(retrieve_node(state))

            yield {
                "type": "stage",
                "stage": "grade",
                "detail": {"retrieved": len(state["retrieved_docs"])},
            }
            state.update(grade_node(state))

            yield {
                "type": "stage",
                "stage": "generate",
                "detail": {
                    "retrieved": len(state["retrieved_docs"]),
                    "kept": len(state["graded_docs"]),
                },
            }

            generation = generate_stream(state)
            try:
                while True:
                    try:
                        text = next(generation)
                    except StopIteration as stop:
                        result = stop.value
                        break
                    parts.append(text)
                    yield {"type": "token", "text": text}
            finally:
                generation.close()
    except GeneratorExit:
        history.cancel(entry_id, "".join(parts))
        raise
    except Exception as exc:
        history.fail(entry_id)
        yield {"type": "error", "detail": str(exc), "id": entry_id}
        return

    answer = result["answer"]
    citations = result.get("citations", [])
    refused = result.get("refused", False)
    confidence = result.get("confidence", 0.0)
    durations = {s["span"]: s["duration_ms"] for s in spans}
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    history.complete(
        entry_id,
        answer=answer,
        citations=citations,
        refused=refused,
        confidence=confidence,
        latency_ms=latency_ms,
        rerank_ms=durations.get("rerank"),
        generate_ms=durations.get("generate"),
    )
    yield {
        "type": "done",
        "id": entry_id,
        "answer": answer,
        "citations": citations,
        "refused": refused,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "rerank_ms": durations.get("rerank"),
        "generate_ms": durations.get("generate"),
        "model": resolved_model,
    }


def main() -> int:
    query = " ".join(sys.argv[1:]) or "What is this document collection about?"

    from src.rag.model_catalog import list_installed

    installed = list_installed()
    if not installed:
        print("no chat model downloaded -- pull one first (see Settings, or "
              "`ollama pull <model>`)")
        return 1
    model = installed[0]

    print(f"query: {query}")
    print(f"model: {model}")
    result = ask(query, model=model)
    print(f"answer: {result['answer']}")
    print(f"confidence: {result['confidence']}")
    citations = format_citations(result["citations"])
    if citations:
        print(citations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
