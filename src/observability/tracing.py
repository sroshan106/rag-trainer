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

_details: contextvars.ContextVar = contextvars.ContextVar("span_details", default=None)
_sinks: contextvars.ContextVar = contextvars.ContextVar("span_sinks", default=())


def tracing_enabled() -> bool:
    return env_flag(TRACE_ENV, default=False)


def configure_logging(level: int = logging.INFO) -> None:
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def detail(**fields) -> None:
    current = _details.get()
    if current is not None:
        current.update(fields)


def _emit(span: dict) -> None:
    for sink in _sinks.get():
        sink.append(span)
    logger.info(json.dumps(span, default=str))


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
