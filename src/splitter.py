"""Phase 3: chunk documents for embedding."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def split_documents(
    docs: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


if __name__ == "__main__":
    from src.loaders import load_documents

    docs = load_documents("data/documents.csv")
    chunks = split_documents(docs)
    print(f"{len(docs)} documents -> {len(chunks)} chunks")
    print("sample chunk:", chunks[0].page_content[:200])
    print("sample metadata:", chunks[0].metadata)
