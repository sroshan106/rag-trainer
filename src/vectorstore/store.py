import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import httpx
import sqlalchemy as sa

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

from src.config import get_settings
from src.db.engine import get_engine

COLLECTION_NAME = "rag_chunks"
EMBED_MODEL = get_settings().embed_model

ProgressHook = Callable[[float, str], None]


class EmbedModelNotInstalled(RuntimeError):
    pass


def ollama_model_names() -> set[str]:
    resp = httpx.get(f"{get_settings().ollama_base_url}/api/tags", timeout=5.0)
    resp.raise_for_status()
    return {m["name"] for m in resp.json().get("models", [])}


def model_is_installed(model: str, names: set[str]) -> bool:
    if model in names:
        return True
    if ":" not in model:
        return any(name.split(":", 1)[0] == model for name in names)
    return False


def _require_embed_model() -> None:
    embed_model = get_settings().embed_model
    if model_is_installed(embed_model, ollama_model_names()):
        return
    raise EmbedModelNotInstalled(
        f"Embedding model {embed_model!r} is not installed. "
        "Download it from Settings > Embeddings before ingesting."
    )


def _embeddings(num_gpu: int | None = None) -> OllamaEmbeddings:
    s = get_settings()
    kwargs = {} if num_gpu is None else {"num_gpu": num_gpu}
    return OllamaEmbeddings(model=s.embed_model, base_url=s.ollama_base_url, **kwargs)


def _embed_batch(
    embeddings: OllamaEmbeddings, texts: list[str], retries: int | None = None
) -> list[list[float]]:
    if retries is None:
        retries = get_settings().embed_retries
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
    connection = connection or get_settings().database_url
    ids = ids or [str(uuid.uuid4()) for _ in chunks]
    if not chunks:
        return ids

    s = get_settings()
    batch_size = batch_size or s.embed_batch_size
    _require_embed_model()
    embeddings = _embeddings(num_gpu=999)
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=connection,
    )

    texts = [chunk.page_content for chunk in chunks]

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
    with ThreadPoolExecutor(max_workers=max(1, s.embed_workers)) as pool:
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
    connection = connection or get_settings().database_url
    return PGVector(
        embeddings=_embeddings(num_gpu=0),
        collection_name=COLLECTION_NAME,
        connection=connection,
    )


def delete_chunks(chunk_ids: list[str], connection: str | None = None) -> None:
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
    engine = get_engine(connection)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                sa.text(_COUNT_SQL), {"collection": COLLECTION_NAME}
            ).first()
        return int(row[0]) if row else 0
    except sa.exc.SQLAlchemyError:
        return 0
