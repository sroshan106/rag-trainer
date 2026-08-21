"""Which models this app can use, and what's actually on disk right now.

Model presence checking and download management for chat, embedding, and reranker models. Chat/embed via Ollama, reranker via HuggingFace Hub.
"""

import json
import os

import httpx

from src.observability.logging import log
from src.rag.nodes import AVAILABLE_MODELS
from src.vectorstore import rerank
from src.vectorstore.store import EMBED_MODEL

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Chat models the UI is allowed to offer as choices for download and selection.
CATALOG = AVAILABLE_MODELS


def _ensure_in_tuple(item: str, defaults: tuple[str, ...]) -> tuple[str, ...]:
    return defaults if item in defaults else (item, *defaults)


RERANK_MODEL = rerank.RERANK_MODEL

_DEFAULT_RERANK_MODELS = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-large",
    "mixedbread-ai/mxbai-rerank-base-v1",
    "jinaai/jina-reranker-v2-base-multilingual",
)
RERANK_CATALOG = _ensure_in_tuple(RERANK_MODEL, _DEFAULT_RERANK_MODELS)

# Minimum hardware requirements, sizes, and architectural metadata for models.
MODEL_METADATA = {
    # Chat models
    "llama3.2:3b": {
        "min_vram": "3 GB",
        "disk_size": "~2.0 GB",
        "params": "3.2B",
        "context": "128k ctx",
        "description": "Fast, balanced general QA with low latency and strong instruction following.",
    },
    "llama3.2:1b": {
        "min_vram": "1.5 GB",
        "disk_size": "~1.3 GB",
        "params": "1.2B",
        "context": "128k ctx",
        "description": "Ultra-lightweight edge model with minimal memory footprint.",
    },
    "qwen3:4b": {
        "min_vram": "3.5 GB",
        "disk_size": "~2.6 GB",
        "params": "4B",
        "context": "32k ctx",
        "description": "Strong reasoning, multilingual support, and coding capabilities.",
    },
    "qwen2.5:3b": {
        "min_vram": "3 GB",
        "disk_size": "~1.9 GB",
        "params": "3.1B",
        "context": "32k ctx",
        "description": "High instruction following, structured output, and coding ability.",
    },
    "gemma2:2b": {
        "min_vram": "2.5 GB",
        "disk_size": "~1.6 GB",
        "params": "2.6B",
        "context": "8k ctx",
        "description": "Google lightweight conversational and knowledge retrieval model.",
    },
    "phi3.5": {
        "min_vram": "3.5 GB",
        "disk_size": "~2.2 GB",
        "params": "3.8B",
        "context": "128k ctx",
        "description": "Microsoft compact reasoning model with high benchmark performance.",
    },
    # Embedding model. Only one is ever active (RAG_EMBED_MODEL) -- the
    # collection's stored vectors are pinned to whichever model wrote them, so
    # there is no catalog of switchable alternatives here, just metadata for
    # whatever EMBED_MODEL currently resolves to.
    "nomic-embed-text": {
        "min_vram": "500 MB",
        "disk_size": "~274 MB",
        "params": "137M",
        "context": "8192 ctx • 768 dim",
        "description": "Default embedding model for vectorstore retrieval with large context window.",
    },
    # Reranker models
    "cross-encoder/ms-marco-MiniLM-L-6-v2": {
        "min_vram": "300 MB",
        "disk_size": "~80 MB",
        "params": "22.7M",
        "context": "512 ctx",
        "description": "Fast cross-encoder reranker for passage re-ranking and noise filtering.",
    },
    "cross-encoder/ms-marco-MiniLM-L-12-v2": {
        "min_vram": "400 MB",
        "disk_size": "~130 MB",
        "params": "33M",
        "context": "512 ctx",
        "description": "12-layer variant of MiniLM offering higher precision while remaining fast.",
    },
    "BAAI/bge-reranker-base": {
        "min_vram": "1.5 GB",
        "disk_size": "~1.1 GB",
        "params": "278M",
        "context": "512 ctx",
        "description": "High-accuracy multilingual cross-encoder reranker (100+ languages).",
    },
    "BAAI/bge-reranker-large": {
        "min_vram": "2.5 GB",
        "disk_size": "~2.2 GB",
        "params": "560M",
        "context": "512 ctx",
        "description": "Top-tier accuracy reranker for demanding retrieval benchmarks.",
    },
    "mixedbread-ai/mxbai-rerank-base-v1": {
        "min_vram": "800 MB",
        "disk_size": "~500 MB",
        "params": "135M",
        "context": "512 ctx",
        "description": "State-of-the-art English reranking precision designed for RAG pipelines.",
    },
    "jinaai/jina-reranker-v2-base-multilingual": {
        "min_vram": "1.2 GB",
        "disk_size": "~560 MB",
        "params": "278M",
        "context": "8192 ctx",
        "description": "Supports long-context reranking up to 8k tokens and multi-language queries.",
    },
}

# Chat models only -- embed and rerank models are not downloadable through the
# UI's pull flow (embed has a single pinned active model; rerank pulls happen
# through pull_reranker's HuggingFace path instead).
_OLLAMA_PULLABLE = CATALOG

# Timeout for reading progress lines from Ollama.
PULL_READ_TIMEOUT_SECONDS = 60.0


def _installed_names() -> set[str]:
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        resp.raise_for_status()
    except httpx.HTTPError:
        # Ollama unreachable -- treat as "nothing installed".
        return set()
    return {m["name"] for m in resp.json().get("models", [])}


def _is_installed(model: str, names: set[str]) -> bool:
    # Fall back to comparing name without tag if catalog entry doesn't specify one.
    if model in names:
        return True
    if ":" not in model:
        return any(name.split(":", 1)[0] == model for name in names)
    return False


def list_installed() -> list[str]:
    """Catalog chat models actually present in the Ollama instance right now."""
    names = _installed_names()
    return [m for m in CATALOG if _is_installed(m, names)]


def embed_installed() -> bool:
    """Whether EMBED_MODEL is present in the Ollama instance right now."""
    return _is_installed(EMBED_MODEL, _installed_names())


def rerankers_installed() -> list[str]:
    """Reranker models from RERANK_CATALOG actually present in local HuggingFace cache."""
    try:
        from huggingface_hub import CacheNotFound, scan_cache_dir
    except ImportError:
        return []
    try:
        cache = scan_cache_dir()
    except CacheNotFound:
        return []
    except Exception as exc:  # noqa: BLE001 - a status check must not 500 the page
        log("warning", "reranker cache probe failed", error=repr(exc))
        return []
    cached_repos = {repo.repo_id for repo in cache.repos}
    return [m for m in RERANK_CATALOG if m in cached_repos]


def reranker_installed(model: str = RERANK_MODEL) -> bool:
    """Whether a specific reranker model is in the local HuggingFace cache."""
    return model in rerankers_installed()


def pull_ollama_model(model: str, on_progress, should_stop=None) -> None:
    """Stream a model pull from Ollama, reporting progress as it downloads.
    Stopping is cooperative; should_stop is polled between streamed lines."""
    if model not in _OLLAMA_PULLABLE and model != EMBED_MODEL:
        raise ValueError(f"{model!r} is not downloadable here: {list(_OLLAMA_PULLABLE)}")

    with httpx.stream(
        "POST",
        f"{OLLAMA_BASE_URL}/api/pull",
        json={"model": model},
        # No overall deadline -- a multi-GB pull legitimately runs for many
        # minutes -- but a per-read one, so a server that stops sending
        # progress lines fails instead of hanging this job forever (a hang is
        # also what made a "cancelled" pull keep running: iter_lines never
        # came back, so the cancel flag was never polled).
        timeout=httpx.Timeout(None, connect=10.0, read=PULL_READ_TIMEOUT_SECONDS),
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if should_stop is not None and should_stop():
                return
            if not line:
                continue
            event = json.loads(line)
            status = event.get("status", "")
            completed = event.get("completed")
            total = event.get("total")
            fraction = completed / total if completed and total else None
            on_progress(fraction, status)


def pull_reranker(model: str = RERANK_MODEL, on_progress=None) -> None:
    """Download a reranker model from HuggingFace Hub and load it."""
    if on_progress:
        on_progress(None, f"downloading {model}")
    rerank.ensure_loaded(model)
    if on_progress:
        on_progress(1.0, "downloaded")


def delete_ollama_model(model: str) -> None:
    """Delete a model from Ollama."""
    resp = httpx.request(
        "DELETE",
        f"{OLLAMA_BASE_URL}/api/delete",
        json={"model": model},
        timeout=30.0,
    )
    resp.raise_for_status()


def delete_reranker(model: str = RERANK_MODEL) -> None:
    """Delete a reranker model from the local HuggingFace cache."""
    try:
        from huggingface_hub import CacheNotFound, scan_cache_dir
    except ImportError:
        return
    try:
        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == model:
                delete_strategy = cache.delete_revisions(*[r.commit_hash for r in repo.revisions])
                delete_strategy.execute()
                break
    except CacheNotFound:
        pass


def delete_model(model: str) -> None:
    """Delete an Ollama or HuggingFace reranker model from local disk."""
    is_reranker = model in RERANK_CATALOG or model == RERANK_MODEL
    if is_reranker:
        delete_reranker(model)
    else:
        delete_ollama_model(model)

