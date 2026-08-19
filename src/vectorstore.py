"""Phase 4: embeddings + pgvector store."""

import os

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

COLLECTION_NAME = "rag_chunks"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def _embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)


def build_vectorstore(chunks: list[Document], connection: str | None = None) -> PGVector:
    connection = connection or os.environ["DATABASE_URL"]
    return PGVector.from_documents(
        documents=chunks,
        embedding=_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=connection,
    )


def load_vectorstore(connection: str | None = None) -> PGVector:
    connection = connection or os.environ["DATABASE_URL"]
    return PGVector(
        embeddings=_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=connection,
    )
