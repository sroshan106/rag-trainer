"""Load a file into LangChain Documents. Dispatches by extension."""

import csv
import json
from pathlib import Path

from langchain_core.documents import Document

csv.field_size_limit(10_000_000)

TEXT_COLUMN = "text"

SUPPORTED_EXTENSIONS = {".csv", ".txt", ".md", ".json", ".jsonl", ".pdf"}

# Column-name guesses, in priority order, for CSVs that don't use our own schema.
TEXT_ALIASES = [
    "text", "content", "body", "passage", "document", "doc",
    "summary", "description", "context", "answer", "chunk", "abstract",
]
SOURCE_ALIASES = ["source_url", "source", "url", "link", "origin"]
INDEX_ALIASES = ["index", "row_index", "id", "row_id", "idx", "doc_id"]

# Columns unlikely to hold free text -- skipped when guessing by content.
_NON_TEXT_HINTS = set(SOURCE_ALIASES) | set(INDEX_ALIASES)

# A plausible text column should average more than this many characters/row.
_MIN_AVG_TEXT_LEN = 20


class UnusableCSV(ValueError):
    """The file cannot be ingested."""


class UnsupportedFileType(ValueError):
    """The file's extension isn't one we know how to load."""


def _pick_column(fieldnames: list[str], aliases: list[str]) -> str | None:
    lower_map = {f.lower(): f for f in fieldnames}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    return None


def _guess_text_column(fieldnames: list[str], sample_rows: list[dict]) -> str | None:
    """No known text column name -- pick whichever column holds the most text."""
    if not sample_rows:
        return None
    candidates = [f for f in fieldnames if f.lower() not in _NON_TEXT_HINTS] or fieldnames

    def avg_len(field: str) -> float:
        lengths = [len(row.get(field) or "") for row in sample_rows]
        return sum(lengths) / len(lengths)

    best = max(candidates, key=avg_len)
    return best if avg_len(best) > _MIN_AVG_TEXT_LEN else None


def _load_csv(path: Path) -> list[Document]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise UnusableCSV("the file is empty")
        fieldnames = reader.fieldnames
        rows = list(reader)

    text_col = _pick_column(fieldnames, TEXT_ALIASES) or _guess_text_column(
        fieldnames, rows[:200]
    )
    if text_col is None:
        found = ", ".join(fieldnames)
        raise UnusableCSV(
            f"could not find a text column -- found: {found}. "
            "Rename the column holding your text to 'text', or use a common "
            "alias like 'content' or 'passage'."
        )
    source_col = _pick_column(fieldnames, SOURCE_ALIASES)
    index_col = _pick_column(fieldnames, INDEX_ALIASES)

    docs: list[Document] = []
    skipped = 0
    for row in rows:
        text = row.get(text_col)
        if not text or not text.strip():
            skipped += 1
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": row.get(source_col) if source_col else None,
                    "row_index": row.get(index_col) if index_col else None,
                },
            )
        )
    if skipped:
        print(f"loaders: skipped {skipped} row(s) with empty text")
    return docs


def _load_text(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    return [Document(page_content=text, metadata={"source": path.name})]


def _load_json_records(path: Path) -> list[Document]:
    """Accept a JSON array of {text, source_url} objects, or plain strings."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise UnusableCSV("the JSON file must contain an array of objects or strings")
    return _records_to_documents(data)


def _load_jsonl(path: Path) -> list[Document]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return _records_to_documents(records)


def _records_to_documents(records: list) -> list[Document]:
    docs: list[Document] = []
    skipped = 0
    for record in records:
        if isinstance(record, str):
            text, source_url = record, None
        elif isinstance(record, dict):
            text = record.get(TEXT_COLUMN)
            source_url = record.get("source_url")
        else:
            skipped += 1
            continue
        if not text or not text.strip():
            skipped += 1
            continue
        docs.append(Document(page_content=text, metadata={"source": source_url}))
    if skipped:
        print(f"loaders: skipped {skipped} record(s) with no usable text")
    return docs


def _load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": path.name, "page": page_number},
                )
            )
    return docs


_LOADERS = {
    ".csv": _load_csv,
    ".txt": _load_text,
    ".md": _load_text,
    ".json": _load_json_records,
    ".jsonl": _load_jsonl,
    ".pdf": _load_pdf,
}


def load_documents(path: str | Path) -> list[Document]:
    path = Path(path)
    ext = path.suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise UnsupportedFileType(
            f"unsupported file type {ext or '(none)'!r} -- "
            f"expected one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return loader(path)
