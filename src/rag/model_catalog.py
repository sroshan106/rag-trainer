"""Which models this app can use, and what's actually on disk right now.

Split from ``src.rag.nodes``/``src.vectorstore.rerank`` deliberately: those
modules own *how* a model is called, this one owns *whether it exists yet* --
the Settings view's download buttons and the Ask/Benchmark pickers both need
"is X on disk" without importing the generation or reranking path.

Three kinds of model live here, all optional to have pulled at any given
moment (nothing is force-downloaded at startup or import time anymore -- see
docker-compose.yml, which used to run a blocking ``ollama pull`` before the
app would even start):

- Chat models (``CATALOG``, via Ollama): interchangeable, selectable per
  query/benchmark run.
- The embedding model (``EMBED_MODELS``, via Ollama): NOT interchangeable --
  every ingested chunk was vectorized with it, and swapping it silently would
  make retrieval compare vectors from two different spaces. Kept out of
  ``CATALOG`` so it can never be offered as a chat-model choice, but it is
  just as optional to download from Settings as everything else here --
  ingestion/query simply degrade to a clear error if it's missing, they don't
  block anything at import time.
- The reranker (``RERANK_MODEL``, via HuggingFace Hub, not Ollama): already
  optional at runtime through ``RAG_RERANK=false``; this module adds a way to
  pre-download it from Settings instead of paying for it lazily on the first
  query that reranks.
"""

import json
import os

import httpx

from src.observability.logging import log
from src.rag.nodes import AVAILABLE_MODELS
from src.vectorstore import rerank
from src.vectorstore.store import EMBED_MODEL

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Chat models the UI is allowed to offer as *choices* for download and
# selection. Deliberately the same list nodes.py already validates queries
# against -- a model pulled outside this set would be selectable nowhere, so
# there is no reason to let the pull endpoint accept an arbitrary name.
CATALOG = AVAILABLE_MODELS

# Embedding models compatible with 4GB VRAM GPU. Exposed in Settings for download.
_DEFAULT_EMBED_MODELS = (
    "nomic-embed-text",
    "all-minilm",
    "bge-m3",
    "bge-small",
    "mxbai-embed-large",
    "snowflake-arctic-embed",
)
EMBED_MODELS = (
    _DEFAULT_EMBED_MODELS
    if EMBED_MODEL in _DEFAULT_EMBED_MODELS
    else (EMBED_MODEL, *_DEFAULT_EMBED_MODELS)
)

RERANK_MODEL = rerank.RERANK_MODEL

# Reranker models exposed in Settings for download and local caching.
_DEFAULT_RERANK_MODELS = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-large",
    "mixedbread-ai/mxbai-rerank-base-v1",
    "jinaai/jina-reranker-v2-base-multilingual",
)
RERANK_CATALOG = (
    _DEFAULT_RERANK_MODELS
    if RERANK_MODEL in _DEFAULT_RERANK_MODELS
    else (RERANK_MODEL, *_DEFAULT_RERANK_MODELS)
)

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
    # Embedding models
    "nomic-embed-text": {
        "min_vram": "500 MB",
        "disk_size": "~274 MB",
        "params": "137M",
        "context": "8192 ctx • 768 dim",
        "description": "Default embedding model for vectorstore retrieval with large context window.",
    },
    "all-minilm": {
        "min_vram": "250 MB",
        "disk_size": "~120 MB",
        "params": "33M",
        "context": "256 ctx • 384 dim",
        "description": "Extremely fast, lightweight sentence embeddings.",
    },
    "bge-m3": {
        "min_vram": "1.5 GB",
        "disk_size": "~1.2 GB",
        "params": "568M",
        "context": "8192 ctx • 1024 dim",
        "description": "Multilingual embeddings supporting dense, sparse, and multi-vector search.",
    },
    "bge-small": {
        "min_vram": "200 MB",
        "disk_size": "~67 MB",
        "params": "24M",
        "context": "512 ctx • 384 dim",
        "description": "Minimal memory footprint embedding model for resource-constrained setups.",
    },
    "mxbai-embed-large": {
        "min_vram": "1.0 GB",
        "disk_size": "~670 MB",
        "params": "335M",
        "context": "512 ctx • 1024 dim",
        "description": "High retrieval accuracy representation model for English corpus.",
    },
    "snowflake-arctic-embed": {
        "min_vram": "1.0 GB",
        "disk_size": "~669 MB",
        "params": "335M",
        "context": "512 ctx • 1024 dim",
        "description": "Enterprise-grade retrieval embedding model by Snowflake.",
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

_OLLAMA_PULLABLE = (*CATALOG, *EMBED_MODELS)

# Ollama emits a progress line every few hundred milliseconds while a pull is
# alive, so this is generous for a healthy download and still bounds how long
# a wedged one can sit there.
PULL_READ_TIMEOUT_SECONDS = 60.0


def _installed_names() -> set[str]:
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        resp.raise_for_status()
    except httpx.HTTPError:
        # Ollama unreachable -- treat as "nothing installed" rather than
        # raising, so callers degrade to an empty list instead of failing.
        return set()
    return {m["name"] for m in resp.json().get("models", [])}


def _is_installed(model: str, names: set[str]) -> bool:
    # Ollama always tags a pulled model (implicitly ":latest" if the pull
    # didn't name one); a catalog entry that omits the tag -- like
    # EMBED_MODEL's bare "nomic-embed-text" -- would never match the literal
    # "nomic-embed-text:latest" Ollama reports, so fall back to comparing the
    # name without its tag when the catalog entry doesn't specify one.
    if model in names:
        return True
    if ":" not in model:
        return any(name.split(":", 1)[0] == model for name in names)
    return False


def list_installed() -> list[str]:
    """Catalog chat models actually present in the Ollama instance right now.

    A catalog entry that hasn't been pulled (or is still pulling) must not
    show up as selectable in Ask/Benchmark -- picking it would 404 mid-query.
    """
    names = _installed_names()
    return [m for m in CATALOG if _is_installed(m, names)]


def embed_models_installed() -> list[str]:
    names = _installed_names()
    return [m for m in EMBED_MODELS if _is_installed(m, names)]


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
    """Whether a specific reranker model is in the local HuggingFace cache.

    Scans the cache rather than instantiating ``CrossEncoder`` -- loading the
    model just to check whether loading it would download something is the
    exact cost this status check exists to avoid paying on every page load.
    """
    return model in rerankers_installed()


def pull_ollama_model(model: str, on_progress, should_stop=None) -> None:
    """Stream a model pull from Ollama, reporting progress as it downloads.

    Ollama's ``/api/pull`` returns newline-delimited JSON, one status object
    per line -- ``{"status": "pulling ...", "completed": N, "total": M}``
    once the manifest resolves, plain status strings before and after. Bytes,
    not layers, are what ``on_progress`` gets: a model is usually one large
    layer, so byte progress is the only granularity worth showing.

    Stopping is cooperative, like the benchmark run: ``should_stop`` is polled
    between streamed lines, so a cancel lands within one progress line, and
    leaving the ``stream`` block is what actually aborts the download.
    """
    if model not in _OLLAMA_PULLABLE:
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
    """Download a reranker model from HuggingFace Hub and load it.

    No byte-level progress here -- ``sentence_transformers.CrossEncoder``
    doesn't expose a hook into the underlying HF download, so this only ever
    reports the two ends of the job. The load is not wasted work: the
    singleton it populates (``rerank._model``) is what real reranking calls
    reuse afterward.
    """
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
    is_reranker = (
        model in RERANK_CATALOG
        or model == RERANK_MODEL
    )
    if is_reranker:
        delete_reranker(model)
    else:
        delete_ollama_model(model)

