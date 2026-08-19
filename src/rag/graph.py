"""Phase 5: state schema, graph wiring, entrypoint."""

import sys

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from src.rag.nodes import generate_node, grade_node, retrieve_node

MAX_RETRIES = 2


class RAGState(TypedDict):
    query: str
    retrieved_docs: list[Document]
    graded_docs: list[Document]
    answer: str
    retry_count: int


def should_retry(state: RAGState) -> str:
    if not state["graded_docs"] and state["retry_count"] < MAX_RETRIES:
        return "retrieve"
    return "generate"


def build_graph():
    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade", should_retry, {"retrieve": "retrieve", "generate": "generate"}
    )
    workflow.add_edge("generate", END)

    return workflow.compile()


graph = build_graph()


def ask(query: str) -> str:
    result = graph.invoke({"query": query, "retry_count": 0})
    return result["answer"]


def main() -> int:
    query = " ".join(sys.argv[1:]) or "What is this document collection about?"
    print(f"query: {query}")
    print(f"answer: {ask(query)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
