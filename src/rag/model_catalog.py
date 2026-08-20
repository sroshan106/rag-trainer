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

# Not a choice: the one embedding model every ingested chunk was vectorized
# with (src.vectorstore.store.COLLECTION_NAME is keyed on it). Exposed
# separately so Settings can show/download it once without offering it as an
# alternative next to the chat models.
EMBED_MODELS = (EMBED_MODEL,)

RERANK_MODEL = rerank.RERANK_MODEL

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


def reranker_installed() -> bool:
    """Whether RERANK_MODEL is already in the local HuggingFace cache.

    Scans the cache rather than instantiating ``CrossEncoder`` -- loading the
    model just to check whether loading it would download something is the
    exact cost this status check exists to avoid paying on every page load.
    """
    try:
        from huggingface_hub import CacheNotFound, scan_cache_dir
    except ImportError:
        # No hub library installed -- nothing could have been cached by it, so
        # "not installed" is the honest answer and there is nothing to report.
        return False
    try:
        cache = scan_cache_dir()
    except CacheNotFound:
        # The expected "never downloaded anything" case: no cache dir yet.
        return False
    except Exception as exc:  # noqa: BLE001 - a status check must not 500 the page
        # An unreadable or corrupt cache is a failed probe, not an absent
        # model. Still reported as "not installed" so Settings renders, but
        # no longer silently -- that swallow hid real breakage.
        log("warning", "reranker cache probe failed", model=RERANK_MODEL, error=repr(exc))
        return False
    return any(repo.repo_id == RERANK_MODEL for repo in cache.repos)


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


def pull_reranker(on_progress) -> None:
    """Download RERANK_MODEL from HuggingFace Hub and load it.

    No byte-level progress here -- ``sentence_transformers.CrossEncoder``
    doesn't expose a hook into the underlying HF download, so this only ever
    reports the two ends of the job. The load is not wasted work: the
    singleton it populates (``rerank._model``) is what real reranking calls
    reuse afterward.
    """
    on_progress(None, f"downloading {RERANK_MODEL}")
    rerank.ensure_loaded()
    on_progress(1.0, "downloaded")
