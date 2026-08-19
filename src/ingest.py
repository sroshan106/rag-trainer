"""Ingest data/documents.csv into the pgvector store. Run: python -m src.ingest"""

from dotenv import load_dotenv

load_dotenv()

from src.loaders import load_documents
from src.splitter import split_documents
from src.vectorstore import build_vectorstore

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
