"""Persistent record of uploaded/ingested files, keyed by content hash.

Kept in the same Postgres instance as query history, in its own table: this
is a record of what was *ingested*, not part of retrieval. The hash is what
lets the upload route recognize "this exact file already went in" and refuse
to store and embed it a second time, regardless of what it was named.

Writing this record must never break an ingest that otherwise succeeded --
failures here are logged and swallowed, same as query history.
"""

import hashlib
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

logger = logging.getLogger("rag.files")

TABLE_NAME = "ingested_files"

_metadata = sa.MetaData()

ingested_files = sa.Table(
    TABLE_NAME,
    _metadata,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
    sa.Column("filename", sa.Text, nullable=False),
    sa.Column("stored_path", sa.Text, nullable=False),
    # Unique, not just indexed: the upload route relies on a lookup by hash
    # to decide whether a file is a duplicate before it queues an ingest.
    sa.Column("sha256", sa.String(64), nullable=False, unique=True, index=True),
    sa.Column("size_bytes", sa.Integer, nullable=False),
    sa.Column("documents", sa.Integer, nullable=True),
    # Filled in once the ingest job finishes -- the vector store's ids for
    # every chunk this file produced, so "clear this file" knows exactly what
    # to delete. Null while the job is still running.
    sa.Column("chunk_ids", JSONB, nullable=True),
)

_engines: dict[str, sa.Engine] = {}
_lock = threading.Lock()


def _engine(connection: str | None = None) -> sa.Engine:
    url = connection or os.environ["DATABASE_URL"]
    engine = _engines.get(url)
    if engine is None:
        with _lock:
            engine = _engines.get(url)
            if engine is None:
                engine = sa.create_engine(url)
                _metadata.create_all(engine, tables=[ingested_files])
                _engines[url] = engine
    return engine


def hash_file(fileobj: BinaryIO, chunk_size: int = 1024 * 1024) -> str:
    """SHA-256 of a whole binary stream, read from wherever the cursor is."""
    digest = hashlib.sha256()
    for chunk in iter(lambda: fileobj.read(chunk_size), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _row_to_dict(row) -> dict:
    chunk_ids = row.chunk_ids
    if isinstance(chunk_ids, str):
        chunk_ids = json.loads(chunk_ids)
    return {
        "id": row.id,
        "created_at": row.created_at.isoformat(),
        "filename": row.filename,
        "stored_path": row.stored_path,
        "sha256": row.sha256,
        "size_bytes": row.size_bytes,
        "documents": row.documents,
        "chunk_ids": chunk_ids,
    }


def find_by_hash(sha256: str, connection: str | None = None) -> dict | None:
    """The earlier ingest of this exact content, if there was one."""
    stmt = sa.select(ingested_files).where(ingested_files.c.sha256 == sha256)
    with _engine(connection).connect() as conn:
        row = conn.execute(stmt).first()
    return _row_to_dict(row) if row else None


def get(entry_id: str, connection: str | None = None) -> dict | None:
    stmt = sa.select(ingested_files).where(ingested_files.c.id == entry_id)
    with _engine(connection).connect() as conn:
        row = conn.execute(stmt).first()
    return _row_to_dict(row) if row else None


def record(
    filename: str,
    stored_path: str | Path,
    sha256: str,
    size_bytes: int,
    documents: int | None = None,
    connection: str | None = None,
) -> str | None:
    """Store one file's provenance. Returns its id, or None if not stored."""
    entry_id = str(uuid.uuid4())
    try:
        with _engine(connection).begin() as conn:
            conn.execute(
                ingested_files.insert().values(
                    id=entry_id,
                    created_at=datetime.now(timezone.utc),
                    filename=filename,
                    stored_path=str(stored_path),
                    sha256=sha256,
                    size_bytes=size_bytes,
                    documents=documents,
                )
            )
    except Exception:  # noqa: BLE001 - provenance is auxiliary, never fatal
        logger.warning("failed to record ingested file", exc_info=True)
        return None
    return entry_id


def set_chunk_ids(entry_id: str, chunk_ids: list[str], connection: str | None = None) -> None:
    """Fill in a file's vector-store ids once its ingest job has finished."""
    try:
        with _engine(connection).begin() as conn:
            conn.execute(
                ingested_files.update()
                .where(ingested_files.c.id == entry_id)
                .values(chunk_ids=list(chunk_ids))
            )
    except Exception:  # noqa: BLE001 - provenance is auxiliary, never fatal
        logger.warning("failed to record chunk ids for %s", entry_id, exc_info=True)


def delete(entry_id: str, connection: str | None = None) -> bool:
    """Remove one file's record. Returns whether a row was actually removed."""
    with _engine(connection).begin() as conn:
        result = conn.execute(ingested_files.delete().where(ingested_files.c.id == entry_id))
    return result.rowcount > 0


def recent(limit: int = 50, connection: str | None = None) -> list[dict]:
    """Most recently ingested files first."""
    stmt = (
        sa.select(ingested_files)
        .order_by(ingested_files.c.created_at.desc())
        .limit(limit)
    )
    with _engine(connection).connect() as conn:
        return [_row_to_dict(row) for row in conn.execute(stmt)]
