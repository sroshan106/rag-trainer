"""Model registry, context sizing, and Ollama chat client management."""

import os
import threading
from langchain_ollama import ChatOllama

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Chat models sized for 4GB VRAM.
AVAILABLE_MODELS = (
    "llama3.2:3b",
    "llama3.2:1b",
    "qwen3:4b",
    "qwen2.5:3b",
    "gemma2:2b",
    "phi3.5",
)

NUM_CTX = int(os.environ.get("RAG_NUM_CTX", "8192"))
QWEN3_NUM_CTX = int(os.environ.get("RAG_NUM_CTX_QWEN3", "3072"))


def num_ctx_for(model: str) -> int:
    """Return context window limit tailored to prevent CPU spilling."""
    return QWEN3_NUM_CTX if model.startswith("qwen3") else NUM_CTX


_llms: dict[str, ChatOllama] = {}
_llm_lock = threading.Lock()


def get_llm(model: str) -> ChatOllama:
    """Lazily instantiate or return cached ChatOllama client."""
    llm = _llms.get(model)
    if llm is None:
        with _llm_lock:
            llm = _llms.get(model)
            if llm is None:
                llm = ChatOllama(
                    model=model,
                    base_url=OLLAMA_BASE_URL,
                    temperature=0,
                    num_ctx=num_ctx_for(model),
                    reasoning=False,
                )
                _llms[model] = llm
    return llm
