"""Structured JSON logging plus an in-process ring buffer for the Logs view.

Mirrors the shape of ``tracing.py``: a stdlib logger with a JSON formatter so
``docker compose logs`` stays greppable, and a bounded ``collections.deque``
the UI can tail without a second write path to Postgres -- anything worth
keeping past the buffer's window is already in ``traces``, not here.
"""

import collections
import json
import logging
import os
import sys
import time

LOGGER_NAME = "rag.app"

# Bounded so the process can run indefinitely without the buffer growing --
# the UI only ever needs the recent tail, not full history.
RING_BUFFER_SIZE = int(os.environ.get("RAG_LOG_BUFFER_SIZE", "1000"))

_ring: collections.deque = collections.deque(maxlen=RING_BUFFER_SIZE)

logger = logging.getLogger(LOGGER_NAME)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, default=str)


class RingBufferHandler(logging.Handler):
    """Appends each formatted record to the shared ring buffer as a dict."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            formatted = json.loads(self.format(record))
        except Exception:  # noqa: BLE001 - a bad log line must never crash logging
            return
        _ring.append(formatted)


def configure_logging(level: int = logging.INFO) -> None:
    """Attach stdout + ring-buffer handlers once. Safe to call repeatedly."""
    if logger.handlers:
        return
    formatter = JsonFormatter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    ring_handler = RingBufferHandler()
    ring_handler.setFormatter(formatter)
    logger.addHandler(ring_handler)

    logger.setLevel(level)
    logger.propagate = False


def log(level: str, message: str, **fields) -> None:
    """Convenience entry point: ``log("info", "ingest started", job_id=...)``."""
    configure_logging()
    logger.log(getattr(logging, level.upper(), logging.INFO), message, extra={"fields": fields})


def tail(limit: int = 200, level: str | None = None, query: str | None = None) -> list[dict]:
    """Most recent buffered log records, optionally filtered.

    Filters match the level view (min-severity) and free-text search the
    Logs view is required to have -- kept here rather than in the route so
    the SSE stream and the plain GET can share the same logic.
    """
    records = list(_ring)
    if level:
        min_no = getattr(logging, level.upper(), 0)
        records = [r for r in records if getattr(logging, r.get("level", "INFO"), 0) >= min_no]
    if query:
        needle = query.lower()
        records = [r for r in records if needle in json.dumps(r).lower()]
    return records[-limit:]
