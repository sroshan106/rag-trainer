"""Phase 2: load a CSV into LangChain Documents."""

import csv
from pathlib import Path

from langchain_core.documents import Document

csv.field_size_limit(10_000_000)


TEXT_COLUMN = "text"


class UnusableCSV(ValueError):
    """The file cannot be ingested. Message is written to be shown to a user."""


def load_documents(path: str | Path) -> list[Document]:
    docs: list[Document] = []
    skipped = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Checked rather than assumed: a CSV without this column silently
        # produced zero documents and an empty ingest that looked successful.
        if reader.fieldnames is None:
            raise UnusableCSV("the file is empty")
        if TEXT_COLUMN not in reader.fieldnames:
            found = ", ".join(reader.fieldnames)
            raise UnusableCSV(
                f"no '{TEXT_COLUMN}' column -- found: {found}. "
                f"Expected columns: {TEXT_COLUMN} (required), source_url, index."
            )
        for row in reader:
            index = row.get("index")
            source_url = row.get("source_url")
            text = row.get("text")
            if not text or not text.strip():
                skipped += 1
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": source_url, "row_index": index},
                )
            )
    if skipped:
        print(f"loaders: skipped {skipped} row(s) with empty text")
    return docs


if __name__ == "__main__":
    docs = load_documents("tests/benchmark/data/documents.csv")
    print(f"loaded {len(docs)} documents")
    if docs:
        print("sample metadata:", docs[0].metadata)
        print("sample content:", docs[0].page_content[:200])
