"""Management of benchmark test suites and custom test files.

Handles both built-in benchmark test files (in tests/benchmark/data) and
user-uploaded custom test sets (in data/benchmark_uploads).
"""

import csv
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import uuid

logger = logging.getLogger("rag.benchmark.files")

BUILTIN_DIR = Path("tests/benchmark/data")
UPLOAD_DIR = Path("data/benchmark_uploads")
MANIFEST_PATH = UPLOAD_DIR / "manifest.json"

QUESTION_COLUMNS = ("question", "query", "prompt", "q", "question_text")
ANSWER_COLUMNS = ("answer", "ground_truth", "reference", "expected", "target", "a", "expected_answer")
DOC_INDEX_COLUMNS = ("document_index", "doc_index", "doc_id", "index", "document_id")


class UnusableTestFile(ValueError):
    """Raised when an uploaded test file cannot be parsed or used for benchmarking."""


def _find_column(fieldnames: list[str], aliases: tuple[str, ...]) -> str | None:
    cleaned = {f.strip().lower(): f for f in fieldnames if f}
    for alias in aliases:
        if alias in cleaned:
            return cleaned[alias]
    return None


def inspect_test_csv(file_content: str | bytes) -> dict:
    """Inspect CSV content and return metadata: questions count, suite_type, and column mapping.

    Raises UnusableTestFile if the CSV is missing required question columns or has no rows.
    """
    if isinstance(file_content, bytes):
        try:
            text = file_content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UnusableTestFile(f"File must be UTF-8 encoded text: {exc}") from exc
    else:
        text = file_content

    if not text.strip():
        raise UnusableTestFile("The file is empty")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise UnusableTestFile("The file contains no headers")

    q_col = _find_column(reader.fieldnames, QUESTION_COLUMNS)
    if not q_col:
        found = ", ".join(repr(f) for f in reader.fieldnames)
        expected = ", ".join(QUESTION_COLUMNS)
        raise UnusableTestFile(
            f"No question column found (looked for one of: {expected}). Found columns: {found}"
        )

    ans_col = _find_column(reader.fieldnames, ANSWER_COLUMNS)
    doc_col = _find_column(reader.fieldnames, DOC_INDEX_COLUMNS)

    count = 0
    non_empty_answers = 0
    for row in reader:
        q_val = row.get(q_col)
        if q_val and q_val.strip():
            count += 1
            if ans_col:
                a_val = row.get(ans_col)
                if a_val and a_val.strip():
                    non_empty_answers += 1

    if count == 0:
        raise UnusableTestFile("The CSV contains no valid rows with questions")

    suite_type = "answerable" if non_empty_answers > 0 else "no_answer"

    return {
        "questions": count,
        "suite_type": suite_type,
        "question_col": q_col,
        "answer_col": ans_col,
        "doc_index_col": doc_col,
    }


def _load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.warning("Failed to load benchmark uploads manifest", exc_info=True)
        return []


def _save_manifest(entries: list[dict]) -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def get_builtin_test_files() -> list[dict]:
    """Return all built-in test files from tests/benchmark/data."""
    results = []
    if not BUILTIN_DIR.exists():
        return results

    # Built-in question files to include (exclude raw documents corpus)
    builtin_names = [
        "single_passage_answer_questions.csv",
        "multi_passage_answer_questions.csv",
        "no_answer_questions.csv",
    ]

    for name in builtin_names:
        path = BUILTIN_DIR / name
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                info = inspect_test_csv(f.read())
            results.append(
                {
                    "id": name,
                    "name": name,
                    "filename": name,
                    "builtin": True,
                    "questions": info["questions"],
                    "suite_type": info["suite_type"],
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        except Exception:
            logger.warning("Failed to inspect built-in test file %s", path, exc_info=True)

    return results


def get_uploaded_test_files() -> list[dict]:
    """Return all uploaded custom test files."""
    manifest = _load_manifest()
    valid_entries = []
    changed = False

    for entry in manifest:
        stored_path = Path(entry.get("stored_path", ""))
        if stored_path.exists():
            valid_entries.append(entry)
        else:
            changed = True

    if changed:
        _save_manifest(valid_entries)

    return valid_entries


def list_test_files() -> list[dict]:
    """List all available benchmark test files (built-in + uploaded)."""
    return get_builtin_test_files() + get_uploaded_test_files()


def save_uploaded_test_file(filename: str, content: bytes) -> dict:
    """Validate and store an uploaded test CSV file."""
    clean_filename = Path(filename or "test_questions.csv").name
    if not clean_filename.lower().endswith(".csv"):
        clean_filename += ".csv"

    info = inspect_test_csv(content)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4().hex[:12]
    target_path = UPLOAD_DIR / f"{file_id}-{clean_filename}"

    with open(target_path, "wb") as f:
        f.write(content)

    entry = {
        "id": file_id,
        "name": clean_filename,
        "filename": clean_filename,
        "builtin": False,
        "questions": info["questions"],
        "suite_type": info["suite_type"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "size_bytes": len(content),
        "stored_path": str(target_path),
    }

    manifest = _load_manifest()
    manifest.append(entry)
    _save_manifest(manifest)

    return entry


def delete_uploaded_test_file(file_id: str) -> dict:
    """Delete an uploaded custom test file by its ID."""
    manifest = _load_manifest()
    found_idx = -1
    for idx, entry in enumerate(manifest):
        if entry["id"] == file_id:
            found_idx = idx
            break

    if found_idx == -1:
        raise KeyError(f"No custom test file found with id {file_id!r}")

    entry = manifest.pop(found_idx)
    stored_path = Path(entry.get("stored_path", ""))
    stored_path.unlink(missing_ok=True)
    _save_manifest(manifest)

    return entry


def resolve_test_file_path(file_id_or_name: str) -> Path:
    """Resolve a test file ID or filename into its actual file Path."""
    # Check if direct path exists
    candidate = Path(file_id_or_name)
    if candidate.is_file():
        return candidate

    # Check built-ins
    builtin_candidate = BUILTIN_DIR / file_id_or_name
    if builtin_candidate.is_file():
        return builtin_candidate

    # Check uploads by id or filename
    manifest = _load_manifest()
    for entry in manifest:
        if entry["id"] == file_id_or_name or entry["filename"] == file_id_or_name:
            stored_path = Path(entry["stored_path"])
            if stored_path.is_file():
                return stored_path

    # Check upload dir directly
    upload_candidate = UPLOAD_DIR / file_id_or_name
    if upload_candidate.is_file():
        return upload_candidate

    raise FileNotFoundError(f"Could not resolve test file: {file_id_or_name!r}")
