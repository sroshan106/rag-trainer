import threading
from langchain_ollama import ChatOllama

from src.config import get_settings

_settings = get_settings()

OLLAMA_BASE_URL = _settings.ollama_base_url

AVAILABLE_MODELS = (
    "llama3.2:3b",
    "llama3.2:1b",
    "qwen2.5:3b",
    "gemma2:2b",
    "phi3.5",
)

NUM_CTX = _settings.num_ctx


def num_ctx_for(model: str) -> int:
    return NUM_CTX


_llms: dict[str, ChatOllama] = {}
_llm_lock = threading.Lock()


def get_llm(model: str) -> ChatOllama:
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
