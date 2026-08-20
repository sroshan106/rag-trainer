"""Phase 5: state schema, graph wiring, entrypoint."""

import sys
import time

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from src.observability import tracing
from src.rag import history
from src.rag.citations import format_sources
from src.rag.nodes import (
    AVAILABLE_MODELS,
    MODEL,
    generate_node,
    generate_stream,
    grade_node,
    retrieve_node,
)


class RAGState(TypedDict):
    query: str
    model: str
    retrieved_docs: list[Document]
    graded_docs: list[Document]
    answer: str
    sources: list[str]


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


def ask(query: str, model: str | None = None) -> dict:
    """Return the answer plus the sources it was grounded in.

    ``model`` picks which of AVAILABLE_MODELS answers the query; omitted or
    unrecognized falls back to the default. ``sources`` is empty when the
    citations extension is disabled. The exchange is also written to the
    query history table unless RAG_HISTORY=false; that write cannot fail the
    call.
    """
    resolved_model = model if model in AVAILABLE_MODELS else MODEL
    started = time.perf_counter()
    entry_id = history.start(query=query, model=resolved_model)
    try:
        # collect() runs independently of RAG_TRACE -- rerank/generate are the
        # two spans worth breaking out for every query, not just traced ones,
        # since they're where a slow query is actually spending its time.
        with tracing.collect() as spans, tracing.span("ask"):
            result = graph.invoke({"query": query, "model": resolved_model})
    except Exception:
        history.fail(entry_id)
        raise

    answer = result["answer"]
    sources = result.get("sources", [])
    durations = {s["span"]: s["duration_ms"] for s in spans}
    history.complete(
        entry_id,
        answer=answer,
        sources=sources,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
        rerank_ms=durations.get("rerank"),
        generate_ms=durations.get("generate"),
    )
    return {"answer": answer, "sources": sources, "id": entry_id}


def ask_stream(query: str, model: str | None = None):
    """Stream the same work ``ask`` does, as a sequence of event dicts.

    Events are ``{"type": ...}`` dicts, in this order:

    * ``stage``  -- a node is starting: retrieve, grade, or generate. Carries
      the counts the previous node produced, so the caller can say "reranked
      20 down to 5" instead of an untimed spinner.
    * ``token``  -- a fragment of the answer.
    * ``done``   -- the finished answer, its sources, and the same latency
      breakdown ``ask`` records.
    * ``error``  -- the query failed; the generator ends after it.

    The graph is stepped node by node here rather than through the compiled
    ``graph.invoke``: the stage boundaries and the partial answer are the whole
    point of this path, and neither survives a single blocking call. The node
    order is the same one ``build_graph`` wires, and the linear graph is what
    makes stepping it by hand safe.

    Closing this generator (client disconnect, or Cancel) stops generation and
    records the run as cancelled with whatever text had already streamed.
    """
    resolved_model = model if model in AVAILABLE_MODELS else MODEL
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
    sources = result.get("sources", [])
    durations = {s["span"]: s["duration_ms"] for s in spans}
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    history.complete(
        entry_id,
        answer=answer,
        sources=sources,
        latency_ms=latency_ms,
        rerank_ms=durations.get("rerank"),
        generate_ms=durations.get("generate"),
    )
    yield {
        "type": "done",
        "id": entry_id,
        "answer": answer,
        "sources": sources,
        "refused": not sources,
        "latency_ms": latency_ms,
        "rerank_ms": durations.get("rerank"),
        "generate_ms": durations.get("generate"),
        "model": resolved_model,
    }


def main() -> int:
    if tracing.tracing_enabled():
        tracing.configure_logging()
    query = " ".join(sys.argv[1:]) or "What is this document collection about?"
    print(f"query: {query}")
    result = ask(query)
    print(f"answer: {result['answer']}")
    sources = format_sources(result["sources"])
    if sources:
        print(sources)
    return 0


if __name__ == "__main__":
    sys.exit(main())
