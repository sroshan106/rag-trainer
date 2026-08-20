"""Routes for adding documents to the system."""

import hashlib
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.api.schemas import IngestFileEntry, JobResponse
from src.ingestion import files as file_history
from src.ingestion.loaders import SUPPORTED_EXTENSIONS, UnsupportedFileType, UnusableCSV, load_documents
from src.ingestion.pipeline import ingest
from src.ingestion.splitter import DEFAULT_SPLITTER, SPLITTERS
from src.jobs.runner import JobAlreadyRunning, ProgressReporter, runner
from src.vectorstore.store import delete_chunks

router = APIRouter(prefix="/api/ingest", tags=["ingest"])

UPLOAD_DIR = Path("data/uploads")

MAX_UPLOAD_BYTES = 200 * 1024 * 1024

JOB_KIND = "ingest"


def _ingest_job(path: str, file_record_id: str, splitter: str):
    def run(reporter: ProgressReporter) -> dict:
        result = ingest(
            path,
            progress=lambda fraction, message: reporter.update(
                progress=fraction, message=message
            ),
            splitter=splitter,
        )
        file_history.set_chunk_ids(file_record_id, result.get("chunk_ids", []))
        return result

    return run


def _submit(path: str, file_record_id: str, splitter: str) -> dict:
    try:
        job = runner.submit_exclusive(JOB_KIND, _ingest_job(path, file_record_id, splitter))
    except JobAlreadyRunning as exc:
        raise HTTPException(
            status_code=409,
            detail=f"an ingest is already running (job {exc.job.id})",
        ) from exc
    return job.to_dict()


@router.get("/splitters")
def list_splitters() -> dict:
    return {"splitters": list(SPLITTERS), "default": DEFAULT_SPLITTER}


@router.post("/upload", response_model=JobResponse, status_code=202)
def upload_and_ingest(
    file: UploadFile = File(...), splitter: str = Form(DEFAULT_SPLITTER)
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
        documents = len(load_documents(target))
    except (UnusableCSV, UnsupportedFileType) as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (UnicodeDecodeError, OSError, json.JSONDecodeError) as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"could not read the file: {exc}") from exc

    if documents == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="the file has no usable text")

    record_id = file_history.record(
        filename=original,
        stored_path=target,
        sha256=sha256,
        size_bytes=size,
        documents=documents,
    )
    try:
        return _submit(str(target), record_id, splitter)
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

    delete_chunks(entry.get("chunk_ids") or [])
    Path(entry["stored_path"]).unlink(missing_ok=True)
    file_history.delete(file_id)

    return {"deleted_chunks": len(entry.get("chunk_ids") or []), "filename": entry["filename"]}


@router.get("/active", response_model=JobResponse | None)
def active_ingest() -> dict | None:
    """Get the active ingest job."""
    job = runner.active(JOB_KIND)
    return job.to_dict() if job else None
