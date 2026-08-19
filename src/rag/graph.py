"""Phase 5: state schema, graph wiring, entrypoint."""

import sys
import time

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from src.observability import tracing
from src.rag import history
from src.rag.citations import format_sources
from src.rag.nodes import AVAILABLE_MODELS, MODEL, generate_node, grade_node, retrieve_node


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
        with tracing.span("ask"):
            result = graph.invoke({"query": query, "model": resolved_model})
    except Exception:
        history.fail(entry_id)
        raise

    answer = result["answer"]
    sources = result.get("sources", [])
    history.complete(
        entry_id,
        answer=answer,
        sources=sources,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    return {"answer": answer, "sources": sources, "id": entry_id}


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
