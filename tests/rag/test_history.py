import pytest
import sqlalchemy as sa

from src.db import engine as db_engine
from src.rag import history


def record(query, answer, citations=(), refused=False, latency_ms=None, model=None,
           confidence=None):
    entry_id = history.start(query, model=model)
    history.complete(
        entry_id,
        answer,
        citations=list(citations),
        refused=refused,
        confidence=confidence,
        latency_ms=latency_ms,
    )
    return entry_id


CITATION = {
    "file_id": "file-1",
    "filename": "corpus.csv",
    "unit_kind": "row",
    "unit_index": 42,
    "label": "row 42",
    "url": None,
}


@pytest.fixture
def db(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'history.db'}"
    monkeypatch.setattr(db_engine, "_engines", {})
    monkeypatch.setattr(db_engine, "_initialized", set())
    for column in (history.query_history.c.sources, history.query_history.c.citations):
        monkeypatch.setattr(column, "type", sa.JSON(), raising=False)
    monkeypatch.setenv("DATABASE_URL", url)
    return url


def test_record_then_read_back(db):
    entry_id = record(
        "who said it?", "Ruby did.", [CITATION], latency_ms=12.5, model="m",
        confidence=0.82,
    )

    entry = history.get(entry_id)
    assert entry["query"] == "who said it?"
    assert entry["answer"] == "Ruby did."
    assert entry["citations"] == [CITATION]
    assert entry["refused"] is False
    assert entry["confidence"] == 0.82
    assert entry["latency_ms"] == 12.5
    assert entry["model"] == "m"


def test_refusal_is_recorded_from_the_flag_not_inferred_from_citations(db):
    entry_id = record("off topic?", "I don't have enough context.", [], refused=True)

    assert history.get(entry_id)["refused"] is True


def test_an_answer_with_no_citable_chunks_is_not_a_refusal(db):
    entry_id = record("who said it?", "Ruby did.", [], refused=False, confidence=0.7)

    entry = history.get(entry_id)
    assert entry["refused"] is False
    assert entry["citations"] == []


def test_recent_returns_newest_first_and_respects_limit(db):
    for i in range(4):
        record(f"q{i}", "a")

    recent = history.recent(limit=2)

    assert len(recent) == 2
    assert [e["query"] for e in recent] == ["q3", "q2"]


def test_disabled_history_records_nothing(db, monkeypatch):
    monkeypatch.setenv("RAG_HISTORY", "false")

    assert record("q", "a", []) is None
    assert history.recent() == []


def test_a_write_failure_does_not_raise(db, monkeypatch, caplog):
    monkeypatch.setattr(
        history, "_engine", lambda connection=None: (_ for _ in ()).throw(OSError("down"))
    )

    assert record("q", "a", []) is None


def test_delete_all_clears_the_table(db):
    record("q1", "a", [])
    record("q2", "a", [])

    assert history.delete_all() == 2
    assert history.recent() == []


def test_get_returns_none_for_an_unknown_id(db):
    assert history.get("no-such-id") is None


def test_cancel_keeps_the_partial_answer(db):
    entry_id = history.start("q", model="m")

    history.cancel(entry_id, "half an ans")

    entry = history.get(entry_id)
    assert entry["status"] == history.STATUS_CANCELLED
    assert entry["answer"] == "half an ans"


def test_cancel_with_nothing_streamed_leaves_no_answer(db):
    entry_id = history.start("q")

    history.cancel(entry_id)

    entry = history.get(entry_id)
    assert entry["status"] == history.STATUS_CANCELLED
    assert entry["answer"] is None


def test_cancel_without_an_entry_is_a_no_op(db):
    assert history.cancel(None, "text") is None


def test_cancel_swallows_a_write_failure(db, monkeypatch):
    entry_id = history.start("q")
    monkeypatch.setattr(
        history, "_engine", lambda connection=None: (_ for _ in ()).throw(OSError("down"))
    )

    assert history.cancel(entry_id, "partial") is None


def test_delete_removes_one_entry(db):
    keep = record("q1", "a", [])
    drop = record("q2", "a", [])

    assert history.delete(drop) == 1
    assert history.get(drop) is None
    assert history.get(keep) is not None


def test_delete_reports_zero_for_an_unknown_id(db):
    assert history.delete("no-such-id") == 0
