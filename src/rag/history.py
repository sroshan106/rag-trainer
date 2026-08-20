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

STATUS_PENDING = "pending"
STATUS_DONE = "done"
STATUS_ERROR = "error"
# The client hung up (or hit Cancel) while the answer was still streaming.
# Distinct from "error": nothing went wrong, the user just stopped caring,
# and whatever tokens had already arrived are kept.
STATUS_CANCELLED = "cancelled"

query_history = sa.Table(
    TABLE_NAME,
    _metadata,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
    sa.Column("query", sa.Text, nullable=False),
    # Null while the row is still pending -- the answer doesn't exist yet.
    sa.Column("answer", sa.Text, nullable=True),
    # JSONB rather than a join table: sources are read back as a whole list,
    # never queried across, and there is no source entity to point at.
    sa.Column("sources", JSONB, nullable=True),
    sa.Column("refused", sa.Boolean, nullable=True),
    sa.Column("latency_ms", sa.Float, nullable=True),
    # Breakout of latency_ms: how much of the total went to the cross-encoder
    # vs. waiting on Ollama generation, so a slow query points at which knob
    # to tune. Null when the respective step didn't run (rerank disabled,
    # refusal path with no generation).
    sa.Column("rerank_ms", sa.Float, nullable=True),
    sa.Column("generate_ms", sa.Float, nullable=True),
    sa.Column("model", sa.String(128), nullable=True),
    sa.Column("status", sa.String(16), nullable=False, server_default=STATUS_DONE),
)

_engines: dict[str, sa.Engine] = {}
_lock = threading.Lock()


def history_enabled() -> bool:
    """Set RAG_HISTORY=false to answer without recording anything."""
    return env_flag("RAG_HISTORY", default=True)


def _migrate(engine: sa.Engine) -> None:
    """Bring a table created before pending rows existed up to date.

    ``create_all`` only creates missing tables, it never alters existing
    ones -- deployments with a live ``query_history`` table need these
    columns/constraints patched in place. Each statement is independently
    safe to re-run.
    """
    statements = [
        "ALTER TABLE query_history ADD COLUMN IF NOT EXISTS status VARCHAR(16) NOT NULL DEFAULT 'done'",
        "ALTER TABLE query_history ALTER COLUMN answer DROP NOT NULL",
        "ALTER TABLE query_history ALTER COLUMN sources DROP NOT NULL",
        "ALTER TABLE query_history ALTER COLUMN refused DROP NOT NULL",
        "ALTER TABLE query_history ADD COLUMN IF NOT EXISTS rerank_ms DOUBLE PRECISION",
        "ALTER TABLE query_history ADD COLUMN IF NOT EXISTS generate_ms DOUBLE PRECISION",
    ]
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(sa.text(statement))


def _engine(connection: str | None = None) -> sa.Engine:
    url = connection or os.environ["DATABASE_URL"]
    engine = _engines.get(url)
    if engine is None:
        with _lock:
            engine = _engines.get(url)
            if engine is None:
                engine = sa.create_engine(url)
                _metadata.create_all(engine, tables=[query_history])
                if engine.dialect.name == "postgresql":
                    _migrate(engine)
                _engines[url] = engine
    return engine


def start(query: str, model: str | None = None, connection: str | None = None) -> str | None:
    """Insert a pending row before the graph runs. Returns its id, or None if not stored.

    Inserted synchronously up front (rather than only at completion) so the
    query is visible -- as pending -- to every tab/client polling history
    while it's still running, not just after it finishes.
    """
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
                    model=model,
                    status=STATUS_PENDING,
                )
            )
    except Exception:  # noqa: BLE001 - history is auxiliary, never fatal
        logger.warning("failed to record pending query history", exc_info=True)
        return None
    return entry_id


def complete(
    entry_id: str,
    answer: str,
    sources: list[str],
    latency_ms: float | None = None,
    rerank_ms: float | None = None,
    generate_ms: float | None = None,
    connection: str | None = None,
) -> None:
    """Fill in a pending row's answer once the graph finishes."""
    if entry_id is None:
        return
    try:
        with _engine(connection).begin() as conn:
            conn.execute(
                query_history.update()
                .where(query_history.c.id == entry_id)
                .values(
                    answer=answer,
                    sources=list(sources),
                    # Derived once at write time: an answer with no surviving
                    # source is the refusal path, and recomputing that from the
                    # answer text later would mean re-parsing prose.
                    refused=not sources,
                    latency_ms=latency_ms,
                    rerank_ms=rerank_ms,
                    generate_ms=generate_ms,
                    status=STATUS_DONE,
                )
            )
    except Exception:  # noqa: BLE001 - history is auxiliary, never fatal
        logger.warning("failed to complete query history", exc_info=True)


def fail(entry_id: str, connection: str | None = None) -> None:
    """Mark a pending row as failed so it doesn't poll as pending forever."""
    if entry_id is None:
        return
    try:
        with _engine(connection).begin() as conn:
            conn.execute(
                query_history.update()
                .where(query_history.c.id == entry_id)
                .values(status=STATUS_ERROR)
            )
    except Exception:  # noqa: BLE001 - history is auxiliary, never fatal
        logger.warning("failed to mark query history as failed", exc_info=True)


def _row_to_dict(row) -> dict:
    sources = row.sources
    if isinstance(sources, str):
        sources = json.loads(sources)
    return {
        "id": row.id,
        "created_at": row.created_at.isoformat(),
        "query": row.query,
        "answer": row.answer,
        "sources": sources or [],
        "refused": row.refused,
        "latency_ms": row.latency_ms,
        "rerank_ms": row.rerank_ms,
        "generate_ms": row.generate_ms,
        "model": row.model,
        "status": row.status,
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


def cancel(entry_id: str, partial_answer: str = "", connection: str | None = None) -> None:
    """Mark a pending row as cancelled, keeping whatever text had streamed.

    The partial answer is stored rather than discarded: a half-finished answer
    is often enough to see whether the query was going anywhere, and throwing
    it away would leave the row indistinguishable from a failure.
    """
    if entry_id is None:
        return
    try:
        with _engine(connection).begin() as conn:
            conn.execute(
                query_history.update()
                .where(query_history.c.id == entry_id)
                .values(answer=partial_answer or None, status=STATUS_CANCELLED)
            )
    except Exception:  # noqa: BLE001 - history is auxiliary, never fatal
        logger.warning("failed to mark query history as cancelled", exc_info=True)


def delete(entry_id: str, connection: str | None = None) -> int:
    """Remove one entry. Returns how many rows were removed (0 or 1)."""
    with _engine(connection).begin() as conn:
        return conn.execute(
            query_history.delete().where(query_history.c.id == entry_id)
        ).rowcount
