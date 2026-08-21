"""Routes for adding documents to the system."""

import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.api.schemas import IngestFileEntry, JobResponse
from src.ingestion import files as file_history
from src.ingestion.loaders import (
    SUPPORTED_EXTENSIONS,
    UnreadableFile,
    UnsupportedFileType,
    load_documents,
)
from src.ingestion.pipeline import ingest
from src.ingestion.splitter import DEFAULT_SPLITTER, SPLITTERS
from src.jobs.runner import JobAlreadyRunning, ProgressReporter, runner
from src.vectorstore.store import delete_chunks

router = APIRouter(prefix="/api/ingest", tags=["ingest"])

UPLOAD_DIR = Path("data/uploads")

MAX_UPLOAD_BYTES = 200 * 1024 * 1024

JOB_KIND = "ingest"


import json


def _ingest_job(
    path: str,
    file_record_id: str,
    splitter: str,
    filename: str,
    index_columns: list[str] | None = None,
    citation_columns: list[str] | None = None,
):
    def run(reporter: ProgressReporter) -> dict:
        ingest_kwargs = {}
        if index_columns is not None:
            ingest_kwargs["index_columns"] = index_columns
        if citation_columns is not None:
            ingest_kwargs["citation_columns"] = citation_columns

        result = ingest(
            path,
            progress=lambda fraction, message: reporter.update(
                progress=fraction, message=message
            ),
            splitter=splitter,
            file_id=file_record_id,
            filename=filename,
            **ingest_kwargs,
        )
        file_history.set_chunk_ids(file_record_id, result.get("chunk_ids", []))
        return result

    return run


def _submit(
    path: str,
    file_record_id: str,
    splitter: str,
    filename: str,
    index_columns: list[str] | None = None,
    citation_columns: list[str] | None = None,
) -> dict:
    try:
        job = runner.submit_exclusive(
            JOB_KIND,
            _ingest_job(
                path,
                file_record_id,
                splitter,
                filename,
                index_columns=index_columns,
                citation_columns=citation_columns,
            ),
        )
    except JobAlreadyRunning as exc:
        raise HTTPException(
            status_code=409,
            detail=f"an ingest is already running (job {exc.job.id})",
        ) from exc
    return job.to_dict()


@router.get("/splitters")
def list_splitters() -> dict:
    return {"splitters": list(SPLITTERS), "default": DEFAULT_SPLITTER}


def _parse_columns(raw: str | None) -> list[str] | None:
    """A JSON array or comma-separated list of column names, from a form field."""
    if not raw:
        return None
    try:
        val = json.loads(raw)
        if isinstance(val, list):
            return [str(x) for x in val if str(x).strip()]
    except Exception:
        pass
    return [x.strip() for x in raw.split(",") if x.strip()]


@router.post("/upload", response_model=JobResponse, status_code=202)
def upload_and_ingest(
    file: UploadFile = File(...),
    splitter: str = Form(DEFAULT_SPLITTER),
    index_columns: str | None = Form(None),
    citation_columns: str | None = Form(None),
) -> dict:
    if splitter not in SPLITTERS:
        raise HTTPException(
            status_code=422,
            detail=f"unknown splitter {splitter!r} -- choose from {list(SPLITTERS)}",
        )
    original = Path(file.filename or "upload").name
    if Path(original).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type -- expected one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    parsed_index_columns = _parse_columns(index_columns)
    parsed_citation_columns = _parse_columns(citation_columns)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / f"{uuid.uuid4().hex}-{original}"
    digest = hashlib.sha256()
    size = 0
    with open(target, "wb") as out:
        while chunk := file.file.read(1024 * 1024):
            out.write(chunk)
            digest.update(chunk)
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                target.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB",
                )
    sha256 = digest.hexdigest()

    duplicate = file_history.find_by_hash(sha256)
    if duplicate is not None:
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=409,
            detail=f"identical file already ingested as {duplicate['filename']!r} "
            f"on {duplicate['created_at']}",
        )

    try:
        doc_kwargs = {}
        if parsed_index_columns is not None:
            doc_kwargs["index_columns"] = parsed_index_columns
        if parsed_citation_columns is not None:
            doc_kwargs["citation_columns"] = parsed_citation_columns
        documents = len(load_documents(target, filename=original, **doc_kwargs))
    except (UnreadableFile, UnsupportedFileType) as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (UnicodeDecodeError, OSError) as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"could not read the file: {exc}") from exc

    if documents == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="the file has no usable text")

    record_kwargs = {"documents": documents}
    if parsed_index_columns is not None:
        record_kwargs["index_columns"] = parsed_index_columns
    if parsed_citation_columns is not None:
        record_kwargs["citation_columns"] = parsed_citation_columns

    record_id = file_history.record(
        filename=original,
        stored_path=target,
        sha256=sha256,
        size_bytes=size,
        **record_kwargs,
    )
    try:
        return _submit(
            str(target),
            record_id,
            splitter,
            original,
            index_columns=parsed_index_columns,
            citation_columns=parsed_citation_columns,
        )
    except HTTPException:
        target.unlink(missing_ok=True)
        file_history.delete(record_id)
        raise


@router.get("/history", response_model=list[IngestFileEntry])
def list_ingested_files(limit: int = 50) -> list[dict]:
    """List ingested files."""
    return file_history.recent(limit)


@router.delete("/files/{file_id}")
def delete_ingested_file(file_id: str) -> dict:
    """Delete an uploaded file."""
    if runner.active(JOB_KIND) is not None:
        raise HTTPException(status_code=409, detail="an ingest is running")

    entry = file_history.get(file_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="no such ingested file")

    chunk_ids = entry.get("chunk_ids") or []
    try:
        delete_chunks(chunk_ids)
    except Exception as exc:
        file_history.logger.warning("Failed to delete chunks for file %s: %s", file_id, exc)

    try:
        Path(entry["stored_path"]).unlink(missing_ok=True)
    except Exception as exc:
        file_history.logger.warning("Failed to unlink stored file %s: %s", entry["stored_path"], exc)

    file_history.delete(file_id)

    return {"deleted_chunks": len(chunk_ids), "filename": entry["filename"]}


@router.get("/active", response_model=JobResponse | None)
def active_ingest() -> dict | None:
    """Get the active ingest job."""
    job = runner.active(JOB_KIND)
    return job.to_dict() if job else None
