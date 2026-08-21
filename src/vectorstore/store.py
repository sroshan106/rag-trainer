"""Phase 4: embeddings + pgvector store."""

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import httpx
import sqlalchemy as sa

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

from src.db.engine import get_engine

COLLECTION_NAME = "rag_chunks"
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Chunks per embedding request. One request for a whole corpus is what killed
# a 40k-chunk ingest: the Ollama model runner died mid-call and took 70 minutes
# of work with it, since nothing had been written yet.
EMBED_BATCH_SIZE = int(os.environ.get("RAG_EMBED_BATCH", "512"))
# Concurrent embedding requests. Worth raising only if Ollama is started with a
# matching OLLAMA_NUM_PARALLEL; otherwise the extra requests just queue.
EMBED_WORKERS = int(os.environ.get("RAG_EMBED_WORKERS", "4"))
EMBED_RETRIES = int(os.environ.get("RAG_EMBED_RETRIES", "3"))

# Called with (fraction_complete, message) as batches land.
ProgressHook = Callable[[float, str], None]


class EmbedModelNotInstalled(RuntimeError):
    """EMBED_MODEL isn't pulled into Ollama yet."""


def _require_embed_model() -> None:
    """Fail fast instead of letting every embed request 404 and retry into
    the same wall. Ingestion should not silently trigger a multi-hundred-MB
    download -- pull EMBED_MODEL explicitly from Settings > Embeddings."""
    resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
    resp.raise_for_status()
    names = {m["name"] for m in resp.json().get("models", [])}
    if EMBED_MODEL in names or any(n.split(":", 1)[0] == EMBED_MODEL for n in names):
        return
    raise EmbedModelNotInstalled(
        f"Embedding model {EMBED_MODEL!r} is not installed. "
        "Download it from Settings > Embeddings before ingesting."
    )


def _embeddings(num_gpu: int | None = None) -> OllamaEmbeddings:
    kwargs = {} if num_gpu is None else {"num_gpu": num_gpu}
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL, **kwargs)


def _embed_batch(
    embeddings: OllamaEmbeddings, texts: list[str], retries: int = EMBED_RETRIES
) -> list[list[float]]:
    """Embed one batch, retrying and then halving it if the runner drops us.

    A dead Ollama runner shows up as a reset connection on the next request, so
    a plain retry usually succeeds. If it keeps failing the batch is split: the
    cause may be one oversized chunk rather than the batch as a whole.
    """
    delay = 1.0
    failure: Exception | None = None
    for attempt in range(retries):
        try:
            return embeddings.embed_documents(texts)
        except Exception as exc:
            failure = exc
            if attempt == retries - 1:
                break
            time.sleep(delay)
            delay *= 2
    if len(texts) <= 1:
        raise failure
    middle = len(texts) // 2
    return _embed_batch(embeddings, texts[:middle], retries) + _embed_batch(
        embeddings, texts[middle:], retries
    )


def build_vectorstore(
    chunks: list[Document],
    connection: str | None = None,
    ids: list[str] | None = None,
    progress: ProgressHook | None = None,
    batch_size: int | None = None,
) -> list[str]:
    """Embed and store ``chunks`` in batches, returning the ids used.

    Ids are generated here (rather than left to PGVector's default) so a
    caller that needs to delete these exact chunks later -- e.g. clearing one
    uploaded file's vectors -- has something to delete by.

    Each batch is written as soon as it is embedded, so a mid-run failure keeps
    everything embedded up to that point instead of discarding the whole job.
    """
    connection = connection or os.environ["DATABASE_URL"]
    ids = ids or [str(uuid.uuid4()) for _ in chunks]
    if not chunks:
        return ids

    batch_size = batch_size or EMBED_BATCH_SIZE
    _require_embed_model()
    embeddings = _embeddings(num_gpu=999)
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=connection,
    )

    texts = [chunk.page_content for chunk in chunks]

    # Corpora like rag-mini-bioasq repeat whole passages across rows. Identical
    # text embeds to an identical vector, so each distinct text is sent to
    # Ollama once and the vector is reused for every chunk that shares it --
    # every duplicate still gets its own stored row, id, and metadata.
    positions: dict[str, list[int]] = {}
    for index, text in enumerate(texts):
        positions.setdefault(text, []).append(index)
    unique_texts = list(positions)

    if progress and len(unique_texts) < len(chunks):
        progress(0.0, f"embedding {len(unique_texts)} unique of {len(chunks)} chunks")

    starts = list(range(0, len(unique_texts), batch_size))

    def embed(start: int) -> list[list[float]]:
        return _embed_batch(embeddings, unique_texts[start : start + batch_size])

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, EMBED_WORKERS)) as pool:
        for start, vectors in zip(starts, pool.map(embed, starts)):
            batch = unique_texts[start : start + batch_size]
            rows = [
                (index, vector)
                for text, vector in zip(batch, vectors)
                for index in positions[text]
            ]
            vectorstore.add_embeddings(
                texts=[texts[index] for index, _vector in rows],
                embeddings=[vector for _index, vector in rows],
                metadatas=[chunks[index].metadata for index, _vector in rows],
                ids=[ids[index] for index, _vector in rows],
            )
            done += len(rows)
            if progress:
                progress(done / len(chunks), f"embedded {done}/{len(chunks)} chunks")
    return ids


def load_vectorstore(connection: str | None = None) -> PGVector:
    """Query-path vectorstore instance (embeds on CPU to preserve GPU VRAM)."""
    connection = connection or os.environ["DATABASE_URL"]
    return PGVector(
        embeddings=_embeddings(num_gpu=0),
        collection_name=COLLECTION_NAME,
        connection=connection,
    )


def delete_chunks(chunk_ids: list[str], connection: str | None = None) -> None:
    """Remove specific stored chunks by id."""
    if not chunk_ids:
        return
    load_vectorstore(connection).delete(ids=chunk_ids)


_COUNT_SQL = """
SELECT count(*)
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON c.uuid = e.collection_id
WHERE c.name = :collection
"""


def count_chunks(connection: str | None = None) -> int:
    """Return total stored chunks count in the collection, or 0 if missing/empty."""
    engine = get_engine(connection)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                sa.text(_COUNT_SQL), {"collection": COLLECTION_NAME}
            ).first()
        return int(row[0]) if row else 0
    except sa.exc.SQLAlchemyError:
        return 0
