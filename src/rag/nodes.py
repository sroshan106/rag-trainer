"""Phase 5: LangGraph nodes — retrieve, grade, generate."""

import os

from langchain_ollama import ChatOllama

from src.rag.prompts import RAG_PROMPT, format_context
from src.vectorstore.store import load_vectorstore

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
RETRIEVE_K = 5

_vectorstore = None
_llm = None


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_vectorstore()
    return _vectorstore


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOllama(model="llama3.2:3b", base_url=OLLAMA_BASE_URL, temperature=0)
    return _llm


def retrieve_node(state: dict) -> dict:
    docs = _get_vectorstore().similarity_search(state["query"], k=RETRIEVE_K)
    return {
        "retrieved_docs": docs,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def grade_node(state: dict) -> dict:
    # Heuristic filter: drop empty/whitespace-only chunks. Swap for an
    # LLM relevance grader if empty/weak retrievals show up in practice.
    graded = [d for d in state["retrieved_docs"] if d.page_content.strip()]
    return {"graded_docs": graded}


def generate_node(state: dict) -> dict:
    if not state["graded_docs"]:
        return {"answer": "I don't have enough context to answer that question."}

    context = format_context(state["graded_docs"])
    prompt = RAG_PROMPT.format(context=context, question=state["query"])
    response = _get_llm().invoke(prompt)
    return {"answer": response.content}
