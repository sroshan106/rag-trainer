"""The Ingest view's backing routes: run the pipeline as a background job.

Upload is the only way in -- there is no server-side path to ingest on
request, so the dashboard can't be pointed at an arbitrary file on the host.
Uploads are refused while an ingest is already running: embedding the same
rows twice appends duplicates to the collection rather than replacing them.
"""

import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.api.schemas import IngestFileEntry, JobResponse
from src.ingestion import files as file_history
from src.ingestion.loaders import UnusableCSV, load_documents
from src.ingestion.pipeline import ingest
from src.ingestion.splitter import DEFAULT_SPLITTER, SPLITTERS
from src.jobs.runner import ProgressReporter, runner
from src.vectorstore.store import delete_chunks

router = APIRouter(prefix="/api/ingest", tags=["ingest"])

# Uploads land beside the bundled corpus, which is already gitignored.
UPLOAD_DIR = Path("data/uploads")

# Generous for a CSV of text, small enough that a mistaken upload cannot fill
# the disk before it is rejected.
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
    running = runner.active(JOB_KIND)
    if running is not None:
        raise HTTPException(
            status_code=409,
            detail=f"an ingest is already running (job {running.id})",
        )
    return runner.submit(JOB_KIND, _ingest_job(path, file_record_id, splitter)).to_dict()


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
    if runner.active(JOB_KIND) is not None:
        raise HTTPException(status_code=409, detail="an ingest is already running")

    # The client's filename is used for display only; the stored name is
    # generated, so a crafted name cannot escape the upload directory.
    original = Path(file.filename or "upload.csv").name
    if not original.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="only .csv files are supported")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / f"{uuid.uuid4().hex}-{original}"
    # Hashed while it's copied rather than re-read afterwards -- one pass over
    # the upload, not two.
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

    # Same bytes already ingested under some name -- keep the earlier copy as
    # the record of truth and refuse to embed the rows a second time.
    duplicate = file_history.find_by_hash(sha256)
    if duplicate is not None:
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=409,
            detail=f"identical file already ingested as {duplicate['filename']!r} "
            f"on {duplicate['created_at']}",
        )

    # Parsed before the job starts so a malformed CSV fails the request the
    # user is watching, rather than a background job they have to go read.
    try:
        documents = len(load_documents(target))
    except UnusableCSV as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (UnicodeDecodeError, OSError) as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"could not read the CSV: {exc}") from exc

    if documents == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="the CSV has no rows with text")

    record_id = file_history.record(
        filename=original,
        stored_path=target,
        sha256=sha256,
        size_bytes=size,
        documents=documents,
    )
    return _submit(str(target), record_id, splitter)


@router.get("/history", response_model=list[IngestFileEntry])
def list_ingested_files(limit: int = 50) -> list[dict]:
    """The provenance/dedup log: what was uploaded, its hash, and where the
    saved copy lives."""
    return file_history.recent(limit)


@router.delete("/files/{file_id}")
def delete_ingested_file(file_id: str) -> dict:
    """Undo one upload: drop its chunks from the vector store, delete the
    saved copy, and remove its record -- a re-upload of the same bytes is no
    longer treated as a duplicate afterwards."""
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
    """Lets the view re-attach to a run in progress after a page reload."""
    job = runner.active(JOB_KIND)
    return job.to_dict() if job else None
