"""Phase 5: node-level tracing, logging, and latency instrumentation.

Emits one structured JSON record per graph node with its duration and the
fields that node chose to expose — retrieval scores, the grading cutoff, the
generation token counts. That record is both the trace and the latency sample,
so there is no second timing path to keep in sync.

Off by default; enable with RAG_TRACE=true. Kept local deliberately: no
LangSmith, no hosted collector, nothing leaves the machine.

Spans can also be captured in-process rather than logged, which is how
``tests/benchmark/run_latency.py`` aggregates percentiles:

    with tracing.collect() as spans:
        ask("some query")
    # spans is a list of {"node": ..., "duration_ms": ..., ...}
"""

import contextvars
import functools
import json
import logging
import sys
import time
from contextlib import contextmanager

from src.config import env_flag

TRACE_ENV = "RAG_TRACE"
LOGGER_NAME = "rag.trace"

logger = logging.getLogger(LOGGER_NAME)

# Fields the current node has attached via detail(). None when no span is open.
_details: contextvars.ContextVar = contextvars.ContextVar("span_details", default=None)
# When a collect() block is active, spans are appended here instead of only logged.
_sink: contextvars.ContextVar = contextvars.ContextVar("span_sink", default=None)


def tracing_enabled() -> bool:
    return env_flag(TRACE_ENV, default=False)


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a stderr handler once. Safe to call repeatedly."""
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def detail(**fields) -> None:
    """Attach fields to the span currently being recorded. No-op when inactive."""
    current = _details.get()
    if current is not None:
        current.update(fields)


def _emit(span: dict) -> None:
    sink = _sink.get()
    if sink is not None:
        sink.append(span)
    logger.info(json.dumps(span, default=str))


@contextmanager
def span(name: str):
    """Record one named span, if tracing is on or a collect() block is active.

    Used directly for spans that are not graph nodes — the end-to-end timing in
    ``ask()``, for instance.
    """
    if not tracing_enabled() and _sink.get() is None:
        yield
        return

    token = _details.set({})
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        fields = _details.get() or {}
        try:
            _details.reset(token)
        except ValueError:
            _details.set(None)
        _emit({"span": name, "duration_ms": duration_ms, **fields})


def traced(name: str):
    """Decorate a graph node so each call records a span.

    Skips instrumentation entirely when tracing is disabled and nothing is
    collecting, so the normal query path pays nothing.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state):
            with span(name):
                return fn(state)

        return wrapper

    return decorator


@contextmanager
def collect():
    """Capture spans in memory for the duration of the block."""
    spans: list[dict] = []
    token = _sink.set(spans)
    try:
        yield spans
    finally:
        try:
            _sink.reset(token)
        except ValueError:
            _sink.set(None)
