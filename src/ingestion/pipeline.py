"""Ingest data/documents.csv into the pgvector store. Run: python -m src.ingestion.pipeline"""

from src.ingestion.loaders import load_documents
from src.ingestion.splitter import split_documents
from src.vectorstore.lexical import ensure_index
from src.vectorstore.store import build_vectorstore

DATA_PATH = "data/documents.csv"


def main() -> None:
    docs = load_documents(DATA_PATH)
    print(f"loaded {len(docs)} documents")

    chunks = split_documents(docs)
    print(f"split into {len(chunks)} chunks")

    build_vectorstore(chunks)

    # Hybrid retrieval's full-text half reads the same rows, so the index is
    # built here rather than in a separate migration step.
    if ensure_index():
        print("built full-text index")
    print("ingest complete")


if __name__ == "__main__":
    main()
