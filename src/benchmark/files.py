import csv
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import uuid

logger = logging.getLogger("rag.benchmark.files")

UPLOAD_DIR = Path("data/benchmark_uploads")
MANIFEST_PATH = UPLOAD_DIR / "manifest.json"

QUESTION_COLUMNS = ("question", "query", "prompt", "q", "question_text")
ANSWER_COLUMNS = ("answer", "ground_truth", "reference", "expected", "target", "a", "expected_answer")
DOC_INDEX_COLUMNS = ("document_index", "doc_index", "doc_id", "index", "document_id")


class UnusableTestFile(ValueError):
    pass


def _find_column(fieldnames: list[str], aliases: tuple[str, ...]) -> str | None:
    cleaned = {f.strip().lower(): f for f in fieldnames if f}
    for alias in aliases:
        if alias in cleaned:
            return cleaned[alias]
    return None


def inspect_test_csv(
    file_content: str | bytes,
    question_col: str | None = None,
    answer_col: str | None = None,
    doc_index_col: str | None = None,
) -> dict:
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

    fieldnames = [f for f in reader.fieldnames if f is not None]

    q_col = None
    if question_col and question_col in fieldnames:
        q_col = question_col
    else:
        q_col = _find_column(fieldnames, QUESTION_COLUMNS)

    if not q_col:
        found = ", ".join(repr(f) for f in fieldnames)
        expected = ", ".join(QUESTION_COLUMNS)
        raise UnusableTestFile(
            f"No question column found (looked for one of: {expected}). Found columns: {found}"
        )

    ans_col = None
    if answer_col is not None:
        if answer_col in fieldnames:
            ans_col = answer_col
    else:
        ans_col = _find_column(fieldnames, ANSWER_COLUMNS)

    doc_col = None
    if doc_index_col is not None:
        if doc_index_col in fieldnames:
            doc_col = doc_index_col
    else:
        doc_col = _find_column(fieldnames, DOC_INDEX_COLUMNS)

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


def get_uploaded_test_files() -> list[dict]:
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


def save_uploaded_test_file(
    filename: str,
    content: bytes,
    question_col: str | None = None,
    answer_col: str | None = None,
    doc_index_col: str | None = None,
) -> dict:
    clean_filename = Path(filename or "test_questions.csv").name
    if not clean_filename.lower().endswith(".csv"):
        clean_filename += ".csv"

    info = inspect_test_csv(
        content,
        question_col=question_col,
        answer_col=answer_col,
        doc_index_col=doc_index_col,
    )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4().hex[:12]
    target_path = UPLOAD_DIR / f"{file_id}-{clean_filename}"

    with open(target_path, "wb") as f:
        f.write(content)

    entry = {
        "id": file_id,
        "name": clean_filename,
        "filename": clean_filename,
        "questions": info["questions"],
        "suite_type": info["suite_type"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "size_bytes": len(content),
        "stored_path": str(target_path),
        "question_col": info["question_col"],
        "answer_col": info["answer_col"],
        "doc_index_col": info["doc_index_col"],
    }

    manifest = _load_manifest()
    manifest.append(entry)
    _save_manifest(manifest)

    return entry


def get_test_file_entry(file_id_or_name_or_path: str) -> dict | None:
    manifest = _load_manifest()
    target_str = str(file_id_or_name_or_path)
    for entry in manifest:
        if (
            entry.get("id") == target_str
            or entry.get("filename") == target_str
            or entry.get("stored_path") == target_str
            or Path(entry.get("stored_path", "")).name == target_str
        ):
            return entry

    return None


def delete_test_file(file_id: str) -> dict:
    manifest = _load_manifest()
    found_idx = -1
    for idx, entry in enumerate(manifest):
        if entry["id"] == file_id or entry.get("filename") == file_id:
            found_idx = idx
            break

    if found_idx == -1:
        raise KeyError(f"No test file found with id {file_id!r}")

    entry = manifest.pop(found_idx)
    stored_path = Path(entry.get("stored_path", ""))
    stored_path.unlink(missing_ok=True)
    _save_manifest(manifest)
    return entry


def resolve_test_file_path(file_id_or_name: str) -> Path:
    candidate = Path(file_id_or_name)
    if candidate.is_file():
        return candidate

    manifest = _load_manifest()
    for entry in manifest:
        if entry["id"] == file_id_or_name or entry["filename"] == file_id_or_name:
            stored_path = Path(entry["stored_path"])
            if stored_path.is_file():
                return stored_path

    upload_candidate = UPLOAD_DIR / file_id_or_name
    if upload_candidate.is_file():
        return upload_candidate

    raise FileNotFoundError(f"Could not resolve test file: {file_id_or_name!r}")
