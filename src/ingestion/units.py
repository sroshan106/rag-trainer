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

URL_COLUMNS = ("source_url", "source", "url", "link", "href")

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

_PARAGRAPH_SEPARATOR = "\n\n"


class UnsupportedFileType(ValueError):
    pass


class UnreadableFile(ValueError):
    pass


@dataclass(frozen=True)
class Unit:

    index: int
    kind: str
    text: str
    url: str | None = None
    key: str | None = None
    fields: dict[str, str] | None = None

    @property
    def label(self) -> str:
        return f"{self.kind} {self.index}"

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "kind": self.kind,
            "text": self.text,
            "url": self.url,
            "key": self.key,
            "fields": self.fields,
            "label": self.label,
        }


def _serialise_row(
    row: dict,
    fieldnames: list[str],
    exclude: set[str] = frozenset(),
    index_columns: list[str] | None = None,
) -> str:
    if index_columns is not None:
        index_set = set(index_columns)
        kept = [name for name in fieldnames if name in index_set]
    else:
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
    lowered = _lowered(row)
    for candidate in URL_COLUMNS:
        value = (lowered.get(candidate) or "").strip()
        if value.startswith(("http://", "https://")):
            return value, _original_name(row, candidate)
    return None, None


def _row_key(row: dict) -> tuple[str | None, str | None]:
    lowered = _lowered(row)
    for candidate in KEY_COLUMNS:
        value = lowered.get(candidate)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            return value, _original_name(row, candidate)
    return None, None


def _explicit_fields(row: dict, citation_columns: list[str]) -> dict[str, str]:
    fields = {}
    for name in citation_columns:
        value = row.get(name)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            fields[name] = value
    return fields


def _structured(
    row: dict, citation_columns: list[str] | None = None
) -> tuple[str | None, str | None, set[str], dict[str, str]]:
    key, key_column = _row_key(row)
    url, url_column = _row_url(row)

    if citation_columns:
        fields = _explicit_fields(row, citation_columns)
        picked_url = next(
            (v for v in fields.values() if v.startswith(("http://", "https://"))),
            None,
        )
        return key, picked_url or url, set(fields), fields

    consumed = {name for name in (key_column, url_column) if name}
    fields = {}
    if key_column:
        fields[key_column] = key
    if url_column:
        fields[url_column] = url
    return key, url, consumed, fields


def _csv_units(
    path: Path,
    index_columns: list[str] | None = None,
    citation_columns: list[str] | None = None,
) -> Iterator[Unit]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return
        fieldnames = [name for name in reader.fieldnames if name is not None]
        for index, row in enumerate(reader, start=1):
            key, url, consumed, fields = _structured(row, citation_columns)
            text = _serialise_row(
                row, fieldnames, exclude=consumed, index_columns=index_columns
            )
            if text:
                yield Unit(
                    index=index,
                    kind=KIND_ROW,
                    text=text,
                    url=url,
                    key=key,
                    fields=fields or None,
                )


def _record_to_text(record) -> tuple[str, str | None, str | None]:
    if isinstance(record, str):
        return record.strip(), None, None
    if isinstance(record, dict):
        fieldnames = [str(k) for k in record]
        flat = {str(k): "" if v is None else str(v) for k, v in record.items()}
        key, url, consumed, _fields = _structured(flat)
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
    text = path.read_text(encoding="utf-8")
    line_number = 1
    for block in text.split(_PARAGRAPH_SEPARATOR):
        if block.strip():
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
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return KIND_PAGE
    if ext in (".csv", ".json"):
        return KIND_ROW
    return KIND_LINE


def iter_units(
    path: str | Path,
    index_columns: list[str] | None = None,
    citation_columns: list[str] | None = None,
) -> Iterator[Unit]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        return _csv_units(
            path, index_columns=index_columns, citation_columns=citation_columns
        )
    reader = _READERS.get(ext)
    if reader is None:
        raise UnsupportedFileType(
            f"unsupported file type {path.suffix or '(none)'!r} -- "
            f"expected one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return reader(path)


def read_units(
    path: str | Path,
    offset: int = 0,
    limit: int = 100,
    citation_columns: list[str] | None = None,
) -> list[Unit]:
    if offset < 0 or limit <= 0:
        return []
    return list(
        itertools.islice(
            iter_units(path, citation_columns=citation_columns), offset, offset + limit
        )
    )


def read_unit(
    path: str | Path, index: int, citation_columns: list[str] | None = None
) -> Unit | None:
    for unit in iter_units(path, citation_columns=citation_columns):
        if unit.index == index:
            return unit
        if unit.index > index:
            return None
    return None


def csv_columns(path: str | Path) -> list[str] | None:
    path = Path(path)
    if path.suffix.lower() != ".csv":
        return None
    with open(path, newline="", encoding="utf-8") as f:
        fieldnames = csv.DictReader(f).fieldnames
    return [name for name in fieldnames if name is not None] if fieldnames else []
