import logging

from src.observability import logging as app_logging


def test_log_appends_to_ring_buffer():
    app_logging.configure_logging()
    app_logging._ring.clear()

    app_logging.log("info", "ingest started", job_id="abc123")

    records = app_logging.tail(limit=10)
    assert len(records) == 1
    assert records[0]["message"] == "ingest started"
    assert records[0]["job_id"] == "abc123"
    assert records[0]["level"] == "INFO"


def test_tail_filters_by_level():
    app_logging.configure_logging()
    app_logging._ring.clear()
    app_logging.log("info", "info message")
    app_logging.log("error", "error message")

    errors_only = app_logging.tail(level="error")

    assert len(errors_only) == 1
    assert errors_only[0]["message"] == "error message"


def test_tail_filters_by_text_query():
    app_logging.configure_logging()
    app_logging._ring.clear()
    app_logging.log("info", "ingest started")
    app_logging.log("info", "query answered")

    matches = app_logging.tail(query="ingest")

    assert len(matches) == 1
    assert matches[0]["message"] == "ingest started"


def test_json_formatter_handles_exc_info():
    formatter = app_logging.JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord(
            "test", logging.ERROR, __file__, 1, "failed", None, __import__("sys").exc_info()
        )
        formatted = formatter.format(record)
    assert "boom" in formatted
