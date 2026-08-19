"""Phase 5: state schema, graph wiring, entrypoint."""

import sys

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from src.rag.citations import format_sources
from src.rag.nodes import generate_node, grade_node, retrieve_node


class RAGState(TypedDict):
    query: str
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


def ask(query: str) -> dict:
    """Return the answer plus the sources it was grounded in.

    ``sources`` is empty when the citations extension is disabled.
    """
    result = graph.invoke({"query": query})
    return {"answer": result["answer"], "sources": result.get("sources", [])}


def main() -> int:
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
