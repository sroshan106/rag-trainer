"""Phase 3: chunk documents for embedding."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def _recursive(chunk_size: int, chunk_overlap: int):
    # Tries paragraph/sentence/word boundaries before falling back to a hard
    # cut -- keeps chunks semantically coherent, which is what the retriever
    # and grader are tuned against.
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )


def _token(chunk_size: int, chunk_overlap: int):
    # Cuts purely on token count with no regard for boundaries -- cheaper and
    # more predictable chunk sizes, at the cost of occasionally splitting
    # mid-sentence.
    return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


SPLITTERS = {"recursive": _recursive, "token": _token}
DEFAULT_SPLITTER = "recursive"


def split_documents(
    docs: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    splitter: str = DEFAULT_SPLITTER,
) -> list[Document]:
    build = SPLITTERS.get(splitter)
    if build is None:
        raise ValueError(f"unknown splitter {splitter!r} -- choose from {list(SPLITTERS)}")
    return build(chunk_size, chunk_overlap).split_documents(docs)


