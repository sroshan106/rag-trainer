"""Ingest a CSV into the pgvector store. Run: python -m src.ingestion.pipeline path"""

import argparse
import sys
from pathlib import Path
from typing import Callable

from src.ingestion.loaders import load_documents
from src.ingestion.splitter import SPLITTERS, DEFAULT_SPLITTER, split_documents
from src.vectorstore.lexical import ensure_index
from src.vectorstore.store import build_vectorstore

# Called with (fraction_complete, message). The API's job runner passes one in
# so the Ingest view can show real stage progress; the CLI passes none.
ProgressHook = Callable[[float, str], None]


def ingest(
    path: str | Path,
    progress: ProgressHook | None = None,
    splitter: str = DEFAULT_SPLITTER,
    file_id: str | None = None,
    filename: str | None = None,
) -> dict:
    """Load, split, and embed path, returning what was written.

    ``file_id``/``filename`` identify the upload these chunks belong to, and
    are stamped onto every chunk so a citation can name the document and open
    it. The CLI leaves them unset -- it ingests paths that were never uploaded.
    """
    def report(fraction: float, message: str) -> None:
        if progress:
            progress(fraction, message)

    report(0.05, f"loading {path}")
    docs = load_documents(path, file_id=file_id, filename=filename)

    report(0.2, f"splitting {len(docs)} documents ({splitter})")
    chunks = split_documents(docs, splitter=splitter)

    report(0.3, f"embedding {len(chunks)} chunks")

    # Embedding is the long pole, so batch progress is mapped across the 0.3 ->
    # 0.95 span rather than leaving the bar parked at 30% for the whole run.
    def embed_progress(fraction: float, message: str) -> None:
        report(0.3 + 0.65 * fraction, message)

    chunk_ids = build_vectorstore(chunks, progress=embed_progress)

    report(0.95, "building full-text index")
    index_built = ensure_index()

    report(1.0, "ingest complete")
    return {
        "path": str(path),
        "documents": len(docs),
        "chunks": len(chunks),
        "chunk_ids": chunk_ids,
        "splitter": splitter,
        "index_built": index_built,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument(
        "--splitter",
        choices=list(SPLITTERS),
        default=DEFAULT_SPLITTER,
        help="chunking algorithm to use",
    )
    args = parser.parse_args(argv)

    result = ingest(args.path, progress=lambda _f, message: print(message), splitter=args.splitter)
    print(f"loaded {result['documents']} documents")
    print(f"split into {result['chunks']} chunks ({result['splitter']})")
    if result["index_built"]:
        print("built full-text index")
    return 0


if __name__ == "__main__":
    sys.exit(main())
