import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from src.db.engine import get_engine

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
    sa.Column("sha256", sa.String(64), nullable=False, unique=True, index=True),
    sa.Column("size_bytes", sa.Integer, nullable=False),
    sa.Column("documents", sa.Integer, nullable=True),
    sa.Column("chunk_ids", sa.JSON().with_variant(JSONB, "postgresql"), nullable=True),
    sa.Column("index_columns", sa.JSON().with_variant(JSONB, "postgresql"), nullable=True),
    sa.Column("citation_columns", sa.JSON().with_variant(JSONB, "postgresql"), nullable=True),
)

_ADDED_COLUMNS = ("index_columns", "citation_columns")


def _create_table(engine: sa.Engine) -> None:
    _metadata.create_all(engine, tables=[ingested_files])
    with engine.begin() as conn:
        try:
            if engine.dialect.name == "postgresql":
                for name in _ADDED_COLUMNS:
                    conn.execute(
                        sa.text(f"ALTER TABLE ingested_files ADD COLUMN IF NOT EXISTS {name} JSONB")
                    )
            elif engine.dialect.name == "sqlite":
                cols = [
                    row[1]
                    for row in conn.execute(sa.text("PRAGMA table_info(ingested_files)"))
                ]
                for name in _ADDED_COLUMNS:
                    if name not in cols:
                        conn.execute(
                            sa.text(f"ALTER TABLE ingested_files ADD COLUMN {name} JSON")
                        )
        except Exception:
            pass


def _engine(connection: str | None = None) -> sa.Engine:
    return get_engine(connection, init=_create_table)


def hash_file(fileobj: BinaryIO, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: fileobj.read(chunk_size), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _row_to_dict(row) -> dict:
    chunk_ids = row.chunk_ids
    if isinstance(chunk_ids, str):
        chunk_ids = json.loads(chunk_ids)
    index_columns = getattr(row, "index_columns", None)
    if isinstance(index_columns, str):
        try:
            index_columns = json.loads(index_columns)
        except Exception:
            pass
    citation_columns = getattr(row, "citation_columns", None)
    if isinstance(citation_columns, str):
        try:
            citation_columns = json.loads(citation_columns)
        except Exception:
            pass
    return {
        "id": row.id,
        "created_at": row.created_at.isoformat(),
        "filename": row.filename,
        "stored_path": row.stored_path,
        "sha256": row.sha256,
        "size_bytes": row.size_bytes,
        "documents": row.documents,
        "chunk_ids": chunk_ids,
        "index_columns": index_columns,
        "citation_columns": citation_columns,
    }


def find_by_hash(sha256: str, connection: str | None = None) -> dict | None:
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
    index_columns: list[str] | None = None,
    citation_columns: list[str] | None = None,
    connection: str | None = None,
) -> str | None:
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
                    index_columns=index_columns,
                    citation_columns=citation_columns,
                )
            )
    except Exception:
        logger.warning("failed to record ingested file", exc_info=True)
        return None
    return entry_id


def set_chunk_ids(entry_id: str, chunk_ids: list[str], connection: str | None = None) -> None:
    try:
        with _engine(connection).begin() as conn:
            conn.execute(
                ingested_files.update()
                .where(ingested_files.c.id == entry_id)
                .values(chunk_ids=list(chunk_ids))
            )
    except Exception:
        logger.warning("failed to record chunk ids for %s", entry_id, exc_info=True)


def delete(entry_id: str, connection: str | None = None) -> bool:
    with _engine(connection).begin() as conn:
        result = conn.execute(ingested_files.delete().where(ingested_files.c.id == entry_id))
    return result.rowcount > 0


def recent(limit: int = 50, connection: str | None = None) -> list[dict]:
    stmt = (
        sa.select(ingested_files)
        .order_by(ingested_files.c.created_at.desc())
        .limit(limit)
    )
    with _engine(connection).connect() as conn:
        return [_row_to_dict(row) for row in conn.execute(stmt)]
