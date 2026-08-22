"""RAG state schema, graph wiring, and the ask/stream/compare entrypoints."""

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
    # Set by the node that actually decides, never inferred from whether any
    # citations came back -- a chunk can be retrieved and used while carrying
    # no usable provenance to cite.
    refused: bool
    confidence: float


def build_graph():
    # Linear retrieve -> grade -> generate. There is deliberately no retry
    # edge: retrieval is deterministic and results come back sorted by
    # descending similarity, so re-running the same query — at any k — can
    # never surface a chunk that clears the grader's cutoff when the top
    # results did not. Reinstate a loop only alongside query rewriting.
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
    """Run retrieve -> grade -> generate, returning (result, span_durations_ms, latency_ms)."""
    started = time.perf_counter()
    # collect() runs independently of RAG_TRACE -- rerank/generate are the
    # two spans worth breaking out for every query, not just traced ones,
    # since they're where a slow query is actually spending its time.
    with tracing.collect() as spans, tracing.span("ask"):
        result = graph.invoke({"query": query, "model": resolved_model})
    durations = {s["span"]: s["duration_ms"] for s in spans}
    return result, durations, round((time.perf_counter() - started) * 1000, 1)


def ask(query: str, model: str | None = None) -> dict:
    """Return the answer plus the citations it was grounded in.

    Model is required. Exchange is written to query history table.
    """
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
    """Grounded answer -- same retrieve/grade/generate path as ``ask`` -- but
    nothing written to query history. Pairs with ``ask_direct`` for the
    Benchmark view's side-by-side comparison, which must not pollute Ask's
    history with ad-hoc test queries.
    """
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
    """Answer with no retrieval step, and nothing written to query history.

    Exists to compare against ``ask``: same model, same generation settings,
    the only difference is that this path never touches the vectorstore. The
    gap between the two answers is what embedding/retrieval contributed.
    """
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
    """Stream the same work ``ask`` does, as a sequence of event dicts.
    
    Yields stages (retrieve, grade, generate), tokens, and done/error events.
    Closing this generator early stops generation and cancels the run.
    """
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
                # Reached on cancellation too, and this is what interrupts
                # Ollama mid-generation rather than leaving it running for an
                # answer nobody is waiting for.
                generation.close()
    except GeneratorExit:
        history.cancel(entry_id, "".join(parts))
        raise
    except Exception as exc:  # noqa: BLE001 - reported to the client as an event
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
    if tracing.tracing_enabled():
        tracing.configure_logging()
    query = " ".join(sys.argv[1:]) or "What is this document collection about?"

    # CLI convenience: picks the first installed model.
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
