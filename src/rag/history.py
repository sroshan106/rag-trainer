"""Persistent record of asked questions, their answers, and their citations.

Kept in the same Postgres instance as the vectors, but in its own table and
its own module: this is a record of what the system was *asked*, not part of
retrieval, and nothing in the query path reads it back.

Writing history must never cost an answer. Every failure here is logged and
swallowed -- a full disk or a dropped connection should degrade the audit
trail, not turn a working query into an error.
"""

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from src.config import env_flag

logger = logging.getLogger("rag.history")

TABLE_NAME = "query_history"

_metadata = sa.MetaData()

query_history = sa.Table(
    TABLE_NAME,
    _metadata,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
    sa.Column("query", sa.Text, nullable=False),
    sa.Column("answer", sa.Text, nullable=False),
    # JSONB rather than a join table: sources are read back as a whole list,
    # never queried across, and there is no source entity to point at.
    sa.Column("sources", JSONB, nullable=False),
    sa.Column("refused", sa.Boolean, nullable=False),
    sa.Column("latency_ms", sa.Float, nullable=True),
    sa.Column("model", sa.String(128), nullable=True),
)

_engines: dict[str, sa.Engine] = {}
_lock = threading.Lock()


def history_enabled() -> bool:
    """Set RAG_HISTORY=false to answer without recording anything."""
    return env_flag("RAG_HISTORY", default=True)


def _engine(connection: str | None = None) -> sa.Engine:
    url = connection or os.environ["DATABASE_URL"]
    engine = _engines.get(url)
    if engine is None:
        with _lock:
            engine = _engines.get(url)
            if engine is None:
                engine = sa.create_engine(url)
                _metadata.create_all(engine, tables=[query_history])
                _engines[url] = engine
    return engine


def record(
    query: str,
    answer: str,
    sources: list[str],
    latency_ms: float | None = None,
    model: str | None = None,
    connection: str | None = None,
) -> str | None:
    """Store one question/answer pair. Returns its id, or None if not stored."""
    if not history_enabled():
        return None

    entry_id = str(uuid.uuid4())
    try:
        with _engine(connection).begin() as conn:
            conn.execute(
                query_history.insert().values(
                    id=entry_id,
                    created_at=datetime.now(timezone.utc),
                    query=query,
                    answer=answer,
                    sources=list(sources),
                    # Derived once at write time: an answer with no surviving
                    # source is the refusal path, and recomputing that from the
                    # answer text later would mean re-parsing prose.
                    refused=not sources,
                    latency_ms=latency_ms,
                    model=model,
                )
            )
    except Exception:  # noqa: BLE001 - history is auxiliary, never fatal
        logger.warning("failed to record query history", exc_info=True)
        return None
    return entry_id


def _row_to_dict(row) -> dict:
    sources = row.sources
    if isinstance(sources, str):
        sources = json.loads(sources)
    return {
        "id": row.id,
        "created_at": row.created_at.isoformat(),
        "query": row.query,
        "answer": row.answer,
        "sources": sources,
        "refused": row.refused,
        "latency_ms": row.latency_ms,
        "model": row.model,
    }


def recent(limit: int = 50, connection: str | None = None) -> list[dict]:
    """Most recently asked questions first."""
    stmt = (
        sa.select(query_history)
        .order_by(query_history.c.created_at.desc())
        .limit(limit)
    )
    with _engine(connection).connect() as conn:
        return [_row_to_dict(row) for row in conn.execute(stmt)]


def get(entry_id: str, connection: str | None = None) -> dict | None:
    stmt = sa.select(query_history).where(query_history.c.id == entry_id)
    with _engine(connection).connect() as conn:
        row = conn.execute(stmt).first()
    return _row_to_dict(row) if row else None


def delete_all(connection: str | None = None) -> int:
    """Clear the history. Returns how many rows were removed."""
    with _engine(connection).begin() as conn:
        return conn.execute(query_history.delete()).rowcount
