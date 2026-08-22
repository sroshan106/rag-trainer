import pytest

from src.observability import tracing
from src.observability.logging import LOGGER_NAME


@pytest.fixture(autouse=True)
def _no_trace_env(monkeypatch):
    monkeypatch.delenv(tracing.TRACE_ENV, raising=False)


def test_tracing_disabled_by_default():
    assert tracing.tracing_enabled() is False


def test_tracing_enabled_by_env(monkeypatch):
    monkeypatch.setenv(tracing.TRACE_ENV, "true")

    assert tracing.tracing_enabled() is True


def test_tracing_disabled_by_falsy_values(monkeypatch):
    for value in ("false", "FALSE", "0", "no", "off", " off "):
        monkeypatch.setenv(tracing.TRACE_ENV, value)

        assert tracing.tracing_enabled() is False, value


def test_collect_captures_spans_even_when_tracing_disabled():
    with tracing.collect() as spans:
        with tracing.span("work"):
            pass

    assert len(spans) == 1
    assert spans[0]["span"] == "work"
    assert spans[0]["duration_ms"] >= 0


def test_span_records_nothing_when_disabled_and_not_collecting(caplog):
    with caplog.at_level("INFO", logger=LOGGER_NAME):
        with tracing.span("work"):
            pass

    assert caplog.records == []


def test_detail_attaches_fields_to_span():
    with tracing.collect() as spans:
        with tracing.span("work"):
            tracing.detail(kept=3, cutoff=0.64)

    assert spans[0]["kept"] == 3
    assert spans[0]["cutoff"] == 0.64


def test_detail_outside_span_is_noop():
    tracing.detail(orphan=True)


def test_span_logs_to_ring_buffer_when_tracing_enabled(monkeypatch, caplog):
    monkeypatch.setenv(tracing.TRACE_ENV, "true")

    with caplog.at_level("INFO", logger=LOGGER_NAME):
        with tracing.span("work"):
            tracing.detail(kept=3)

    assert len(caplog.records) == 1
    assert caplog.records[0].fields == {"span": "work", "duration_ms": pytest.approx(0, abs=1000), "kept": 3}


def test_nested_spans_keep_details_separate():
    with tracing.collect() as spans:
        with tracing.span("outer"):
            tracing.detail(level="outer")
            with tracing.span("inner"):
                tracing.detail(level="inner")

    by_name = {s["span"]: s for s in spans}
    assert by_name["inner"]["level"] == "inner"
    assert by_name["outer"]["level"] == "outer"


def test_span_records_even_when_body_raises():
    with tracing.collect() as spans:
        with pytest.raises(ValueError):
            with tracing.span("boom"):
                raise ValueError("nope")

    assert spans[0]["span"] == "boom"


def test_traced_decorator_wraps_node_and_preserves_return():
    @tracing.traced("node")
    def node(state):
        tracing.detail(seen=state["x"])
        return {"y": state["x"] * 2}

    with tracing.collect() as spans:
        result = node({"x": 21})

    assert result == {"y": 42}
    assert spans[0]["span"] == "node"
    assert spans[0]["seen"] == 21


def test_traced_decorator_preserves_function_name():
    @tracing.traced("node")
    def my_node(state):
        return {}

    assert my_node.__name__ == "my_node"
