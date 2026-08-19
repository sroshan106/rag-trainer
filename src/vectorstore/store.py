"""Phase 4: embeddings + pgvector store."""

import os
import uuid

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

COLLECTION_NAME = "rag_chunks"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def _embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def build_vectorstore(
    chunks: list[Document], connection: str | None = None, ids: list[str] | None = None
) -> list[str]:
    """Embed and store ``chunks``, returning the id each one was stored under.

    Ids are generated here (rather than left to PGVector's default) so a
    caller that needs to delete these exact chunks later -- e.g. clearing one
    uploaded file's vectors -- has something to delete by.
    """
    connection = connection or os.environ["DATABASE_URL"]
    ids = ids or [str(uuid.uuid4()) for _ in chunks]
    PGVector.from_documents(
        documents=chunks,
        embedding=_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=connection,
        ids=ids,
    )
    return ids


def load_vectorstore(connection: str | None = None) -> PGVector:
    connection = connection or os.environ["DATABASE_URL"]
    return PGVector(
        embeddings=_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=connection,
    )


def delete_chunks(chunk_ids: list[str], connection: str | None = None) -> None:
    """Remove specific stored chunks by id, e.g. everything one file added."""
    if not chunk_ids:
        return
    load_vectorstore(connection).delete(ids=chunk_ids)
