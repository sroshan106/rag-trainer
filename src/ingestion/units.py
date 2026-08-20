"""Addressable units of an uploaded file.

A file is a sequence of units, each with a stable index. That index is what a
citation points at and what the document viewer scrolls to, so it has exactly
one definition -- here -- and both the loader and the viewer read it from this
module. If ingestion and display ever disagreed about what "row 42" means, a
citation would point at text the model never saw, which is worse than showing
no citation at all.

Unit kinds are per-format and chosen to match what a person sees when they open
the file elsewhere:

    csv    row   1-based, header excluded -- the number a spreadsheet shows
    json   row   1-based position in the top-level array
    jsonl  line  1-based physical line number
    txt    line  1-based line number of the unit's first line
    md     line  same as txt
    pdf    page  1-based page number

No file needs a particular schema. A CSV row is serialised whole rather than
having one "text" column picked out of it, so an arbitrary upload works without
being reshaped first.
"""

import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

csv.field_size_limit(10_000_000)

SUPPORTED_EXTENSIONS = {".csv", ".txt", ".md", ".json", ".jsonl", ".pdf"}

KIND_ROW = "row"
KIND_LINE = "line"
KIND_PAGE = "page"

# Column names that hold a link rather than content. Used only to offer a
# better citation target when the uploaded data happens to carry one -- the
# column is still embedded like any other, since a URL is occasionally the
# answer to a question about the corpus.
URL_COLUMNS = ("source_url", "source", "url", "link", "href")

# Column names that hold the dataset's own identifier for a record.
#
# This is deliberately NOT the same thing as ``Unit.index``. The index is where
# a record sits in the file, which is what a citation needs in order to seek
# back to it. The key is what the dataset calls that record, which is what an
# evaluation set's answer sheet refers to. They are routinely different: in
# rag-mini-bioasq the passage with id 31776899 is the 37946th row, and in the
# built-in benchmark corpus the ``index`` column starts at 0 while rows start
# at 1. Storing one in place of the other reads as a plausible number and
# quietly destroys recall scoring, so both are kept.
#
# Ordered most- to least-specific. Identifier columns only -- nothing here
# influences what gets embedded.
KEY_COLUMNS = (
    "document_index",
    "document_id",
    "passage_id",
    "doc_id",
    "row_id",
    "index",
    "id",
    "_id",
    "idx",
)

# Unstructured text is read as one unit per paragraph rather than one unit for
# the whole file: a 50MB .txt as a single unit would be useless to cite and
# useless to page through.
_PARAGRAPH_SEPARATOR = "\n\n"


class UnsupportedFileType(ValueError):
    """The file's extension isn't one we know how to read."""


class UnreadableFile(ValueError):
    """The file's extension is known but its contents cannot be parsed."""


@dataclass(frozen=True)
class Unit:
    """One addressable piece of a file."""

    index: int
    kind: str
    text: str
    # Present only when the unit carried an explicit link of its own.
    url: str | None = None
    # The dataset's own identifier for this record, when it declared one.
    # See KEY_COLUMNS -- this is not interchangeable with ``index``.
    key: str | None = None

    @property
    def label(self) -> str:
        """How this unit is named in a citation, e.g. ``row 42``."""
        return f"{self.kind} {self.index}"

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "kind": self.kind,
            "text": self.text,
            "url": self.url,
            "key": self.key,
            "label": self.label,
        }


def _serialise_row(row: dict, fieldnames: list[str], exclude: set[str] = frozenset()) -> str:
    """Render a whole CSV/JSON row as text, without picking a "content" column.

    A single-column file emits the bare value: labelling it would put the
    header in front of every chunk for no gain. Anything wider keeps
    ``header: value`` lines so the embedding retains which field said what.

    ``exclude`` drops the columns already lifted out as structured metadata --
    the identifier and the link. Embedding those would spend the chunk's budget
    on text nobody can ask a question about; the benchmark corpus carries
    130-character Dropbox URLs that would otherwise lead every first chunk.
    They stay available on the Unit, and the viewer still shows them.
    """
    kept = [name for name in fieldnames if name not in exclude]
    present = [
        (name, value)
        for name, value in ((name, (row.get(name) or "").strip()) for name in kept)
        if value
    ]
    if not present:
        return ""
    if len(kept) == 1:
        return present[0][1]
    return "\n".join(f"{name}: {value}" for name, value in present)


def _lowered(row: dict) -> dict:
    return {str(k).lower(): v for k, v in row.items() if k}


def _original_name(row: dict, lowered_name: str) -> str | None:
    return next((str(k) for k in row if str(k).lower() == lowered_name), None)


def _row_url(row: dict) -> tuple[str | None, str | None]:
    """The row's link. Returns ``(url, the column it came from)``."""
    lowered = _lowered(row)
    for candidate in URL_COLUMNS:
        value = (lowered.get(candidate) or "").strip()
        if value.startswith(("http://", "https://")):
            return value, _original_name(row, candidate)
    return None, None


def _row_key(row: dict) -> tuple[str | None, str | None]:
    """The dataset's identifier. Returns ``(key, the column it came from)``."""
    lowered = _lowered(row)
    for candidate in KEY_COLUMNS:
        value = lowered.get(candidate)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            return value, _original_name(row, candidate)
    return None, None


def _structured(row: dict) -> tuple[str | None, str | None, set[str]]:
    """Pull the key and url out of a row, naming the columns they consumed."""
    key, key_column = _row_key(row)
    url, url_column = _row_url(row)
    consumed = {name for name in (key_column, url_column) if name}
    return key, url, consumed


def _csv_units(path: Path) -> Iterator[Unit]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return
        fieldnames = [name for name in reader.fieldnames if name is not None]
        for index, row in enumerate(reader, start=1):
            key, url, consumed = _structured(row)
            text = _serialise_row(row, fieldnames, exclude=consumed)
            if text:
                yield Unit(index=index, kind=KIND_ROW, text=text, url=url, key=key)


def _record_to_text(record) -> tuple[str, str | None, str | None]:
    """Flatten one JSON record. Returns ``(text, url, key)``."""
    if isinstance(record, str):
        return record.strip(), None, None
    if isinstance(record, dict):
        fieldnames = [str(k) for k in record]
        flat = {str(k): "" if v is None else str(v) for k, v in record.items()}
        key, url, consumed = _structured(flat)
        return _serialise_row(flat, fieldnames, exclude=consumed), url, key
    return str(record).strip(), None, None


def _json_units(path: Path) -> Iterator[Unit]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise UnreadableFile(f"invalid JSON: {exc}") from exc
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise UnreadableFile("the JSON file must hold an array or an object")
    for index, record in enumerate(data, start=1):
        text, url, key = _record_to_text(record)
        if text:
            yield Unit(index=index, kind=KIND_ROW, text=text, url=url, key=key)


def _jsonl_units(path: Path) -> Iterator[Unit]:
    with open(path, encoding="utf-8") as f:
        # Numbered by physical line so the index matches what an editor shows,
        # including for blank lines that yield no unit.
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise UnreadableFile(f"invalid JSON on line {line_number}: {exc}") from exc
            text, url, key = _record_to_text(record)
            if text:
                yield Unit(
                    index=line_number, kind=KIND_LINE, text=text, url=url, key=key
                )


def _text_units(path: Path) -> Iterator[Unit]:
    """One unit per paragraph, indexed by the line the paragraph starts on."""
    text = path.read_text(encoding="utf-8")
    line_number = 1
    for block in text.split(_PARAGRAPH_SEPARATOR):
        if block.strip():
            # Leading blank lines belong to the gap, not the paragraph.
            offset = len(block) - len(block.lstrip("\n"))
            yield Unit(
                index=line_number + offset,
                kind=KIND_LINE,
                text=block.strip(),
            )
        line_number += block.count("\n") + _PARAGRAPH_SEPARATOR.count("\n")


def _pdf_units(path: Path) -> Iterator[Unit]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(str(path))
        pages = list(reader.pages)
    except PdfReadError as exc:
        raise UnreadableFile(f"could not read the PDF: {exc}") from exc

    for page_number, page in enumerate(pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            yield Unit(index=page_number, kind=KIND_PAGE, text=text)


_READERS = {
    ".csv": _csv_units,
    ".txt": _text_units,
    ".md": _text_units,
    ".json": _json_units,
    ".jsonl": _jsonl_units,
    ".pdf": _pdf_units,
}


def is_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def unit_kind(path: str | Path) -> str:
    """The kind of unit this file is addressed by, without reading it."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return KIND_PAGE
    if ext in (".csv", ".json"):
        return KIND_ROW
    return KIND_LINE


def iter_units(path: str | Path) -> Iterator[Unit]:
    """Yield every non-empty unit of ``path``, in file order.

    Streams wherever the format allows, so a large CSV or JSONL never has to be
    held in memory whole.
    """
    path = Path(path)
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise UnsupportedFileType(
            f"unsupported file type {path.suffix or '(none)'!r} -- "
            f"expected one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return reader(path)


def read_units(path: str | Path, offset: int = 0, limit: int = 100) -> list[Unit]:
    """A window of units, sliced by position in the file.

    ``offset`` counts units from the start, which is not the same as a unit's
    ``index`` -- rows that yielded nothing leave gaps in the numbering.
    """
    if offset < 0 or limit <= 0:
        return []
    return list(itertools.islice(iter_units(path), offset, offset + limit))


def read_unit(path: str | Path, index: int) -> Unit | None:
    """The single unit numbered ``index``, or None if the file has no such unit.

    Formats are scanned rather than seeked. Reaching a late row of a large CSV
    means parsing everything before it, which costs well under a second even on
    a 60MB corpus and avoids maintaining an offset index that a re-upload would
    have to invalidate. Revisit only if measurement says so.
    """
    for unit in iter_units(path):
        if unit.index == index:
            return unit
        # Indexes increase monotonically, so passing the target means it is
        # absent -- an empty row that produced no unit, or out of range.
        if unit.index > index:
            return None
    return None


def csv_columns(path: str | Path) -> list[str] | None:
    """Header names, for a CSV the viewer wants to render as a table."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        return None
    with open(path, newline="", encoding="utf-8") as f:
        fieldnames = csv.DictReader(f).fieldnames
    return [name for name in fieldnames if name is not None] if fieldnames else []
