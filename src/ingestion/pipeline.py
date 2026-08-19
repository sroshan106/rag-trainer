"""Ingest data/documents.csv into the pgvector store. Run: python -m src.ingestion.pipeline"""

from src.ingestion.loaders import load_documents
from src.ingestion.splitter import split_documents
from src.vectorstore.store import build_vectorstore

DATA_PATH = "data/documents.csv"


def main() -> None:
    docs = load_documents(DATA_PATH)
    print(f"loaded {len(docs)} documents")

    chunks = split_documents(docs)
    print(f"split into {len(chunks)} chunks")

    build_vectorstore(chunks)
    print("ingest complete")


if __name__ == "__main__":
    main()
