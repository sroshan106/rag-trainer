import json

import httpx

from src.config import get_settings
from src.rag.models import AVAILABLE_MODELS
from src.vectorstore import rerank
from src.vectorstore.store import EMBED_MODEL, model_is_installed, ollama_model_names

OLLAMA_BASE_URL = get_settings().ollama_base_url

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

MODEL_METADATA = {
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
    "nomic-embed-text": {
        "min_vram": "500 MB",
        "disk_size": "~274 MB",
        "params": "137M",
        "context": "8192 ctx • 768 dim",
        "description": "Default embedding model for vectorstore retrieval with large context window.",
    },
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

_OLLAMA_PULLABLE = CATALOG

PULL_READ_TIMEOUT_SECONDS = 60.0


def _installed_names() -> set[str]:
    try:
        return ollama_model_names()
    except httpx.HTTPError:
        return set()


def list_installed() -> list[str]:
    names = _installed_names()
    return [m for m in CATALOG if model_is_installed(m, names)]


def embed_installed() -> bool:
    return model_is_installed(EMBED_MODEL, _installed_names())


def rerankers_installed() -> list[str]:
    cached = rerank.hf_cached_repos()
    return [m for m in RERANK_CATALOG if m in cached]


def reranker_installed(model: str = RERANK_MODEL) -> bool:
    return model in rerankers_installed()


def pull_ollama_model(model: str, on_progress, should_stop=None) -> None:
    if model not in _OLLAMA_PULLABLE and model != EMBED_MODEL:
        raise ValueError(f"{model!r} is not downloadable here: {list(_OLLAMA_PULLABLE)}")

    with httpx.stream(
        "POST",
        f"{OLLAMA_BASE_URL}/api/pull",
        json={"model": model},
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
    if on_progress:
        on_progress(None, f"downloading {model}")
    rerank.ensure_loaded(model)
    if on_progress:
        on_progress(1.0, "downloaded")


def delete_ollama_model(model: str) -> None:
    resp = httpx.request(
        "DELETE",
        f"{OLLAMA_BASE_URL}/api/delete",
        json={"model": model},
        timeout=30.0,
    )
    resp.raise_for_status()


def delete_reranker(model: str = RERANK_MODEL) -> None:
    if model not in rerank.hf_cached_repos():
        return
    from huggingface_hub import scan_cache_dir

    cache = scan_cache_dir()
    for repo in cache.repos:
        if repo.repo_id == model:
            cache.delete_revisions(*[r.commit_hash for r in repo.revisions]).execute()
            break


def delete_model(model: str) -> None:
    is_reranker = model in RERANK_CATALOG or model == RERANK_MODEL
    if is_reranker:
        delete_reranker(model)
    else:
        delete_ollama_model(model)
