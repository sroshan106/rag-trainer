"""Read back the documents that were ingested.

Citations name a file and a unit inside it; this is what lets the UI show that
unit. What is served is the text as the ingest pipeline read it, not a
rendering of the original file -- for a PDF that means the extracted page text
the embedding was actually built from. Showing the model's view is the point:
a citation the reader cannot check against what was retrieved is decorative.
"""

from contextlib import contextmanager
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from src.api.routes import ingest as ingest_route
from src.api.schemas import DocumentMeta, UnitEntry
from src.ingestion import files as file_history
from src.ingestion.units import (
    UnreadableFile,
    UnsupportedFileType,
    csv_columns,
    is_supported,
    read_unit,
    read_units,
    unit_kind,
)

router = APIRouter(prefix="/api/documents", tags=["documents"])

MAX_PAGE = 200


def _stored_path(file_id: str) -> tuple[Path, dict]:
    """Resolve a file id to a path on disk, or raise.

    The path comes out of our own database, but it is still confined to the
    upload directory before anything opens it: a corrupted or hand-edited
    record would otherwise turn this endpoint into an arbitrary file read.
    """
    entry = file_history.get(file_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="no such document")

    root = Path(ingest_route.UPLOAD_DIR).resolve()
    path = Path(entry["stored_path"]).resolve()
    if not path.is_relative_to(root):
        raise HTTPException(status_code=403, detail="document is outside the upload directory")
    if not path.is_file():
        raise HTTPException(status_code=410, detail="the stored copy of this document is gone")
    return path, entry


@contextmanager
def _guarded():
    """Translate reader failures into responses rather than 500s."""
    try:
        yield
    except UnsupportedFileType as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except (UnreadableFile, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"could not read the document: {exc}") from exc


@router.get("/{file_id}", response_model=DocumentMeta)
def document_meta(file_id: str) -> dict:
    """What the viewer needs before it asks for any text."""
    path, entry = _stored_path(file_id)
    if not is_supported(path):
        raise HTTPException(status_code=415, detail="unsupported document type")

    try:
        columns = csv_columns(path)
    except (OSError, UnicodeDecodeError):
        # Only affects table rendering; the units still read fine.
        columns = None

    return {
        "id": entry["id"],
        "filename": entry["filename"],
        "extension": path.suffix.lower(),
        "unit_kind": unit_kind(path),
        # Counted at ingest. Recounting would mean re-parsing the whole file on
        # every page load, which on a 60MB corpus is seconds of work to
        # reproduce a number already stored.
        "units": entry.get("documents"),
        "size_bytes": entry["size_bytes"],
        "created_at": entry["created_at"],
        "chunks": len(entry.get("chunk_ids") or []),
        "columns": columns,
    }


@router.get("/{file_id}/units", response_model=list[UnitEntry])
def document_units(
    file_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=MAX_PAGE),
) -> list[dict]:
    """A window of the document, sliced by position."""
    path, _ = _stored_path(file_id)
    with _guarded():
        return [unit.to_dict() for unit in read_units(path, offset=offset, limit=limit)]


@router.get("/{file_id}/units/{index}", response_model=UnitEntry)
def document_unit(file_id: str, index: int) -> dict:
    """One unit, addressed the way a citation addresses it."""
    path, _ = _stored_path(file_id)
    with _guarded():
        unit = read_unit(path, index)
    if unit is None:
        raise HTTPException(
            status_code=404, detail=f"this document has no {unit_kind(path)} {index}"
        )
    return unit.to_dict()
