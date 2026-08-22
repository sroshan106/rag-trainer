import contextvars
import functools
import time
from contextlib import contextmanager

from src.config import env_flag
from src.observability.logging import log

TRACE_ENV = "RAG_TRACE"

_details: contextvars.ContextVar = contextvars.ContextVar("span_details", default=None)
_sinks: contextvars.ContextVar = contextvars.ContextVar("span_sinks", default=())


def tracing_enabled() -> bool:
    return env_flag(TRACE_ENV, default=False)


def detail(**fields) -> None:
    current = _details.get()
    if current is not None:
        current.update(fields)


def _emit(span: dict) -> None:
    for sink in _sinks.get():
        sink.append(span)
    # Ring-buffered (and so visible to GET /api/metrics/logs) only when
    # tracing is opted into -- the sinks above already give ask() its
    # per-stage durations unconditionally, this is the verbose field dump.
    if tracing_enabled():
        log("info", "trace", **span)


@contextmanager
def span(name: str):
    if not tracing_enabled() and not _sinks.get():
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
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state):
            with span(name):
                return fn(state)

        return wrapper

    return decorator


@contextmanager
def collect():
    spans: list[dict] = []
    token = _sinks.set(_sinks.get() + (spans,))
    try:
        yield spans
    finally:
        try:
            _sinks.reset(token)
        except ValueError:
            _sinks.set(())
