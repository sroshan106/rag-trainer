"""Query-history behaviour, exercised against SQLite rather than Postgres.

The table is created from the same SQLAlchemy metadata either way, so the
insert/read path is real; only JSONB degrades to JSON. What matters most here
is the failure policy: recording must never be able to break a query.
"""

import pytest
import sqlalchemy as sa

from src.rag import history


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A throwaway database, with the module's engine cache cleared around it."""
    url = f"sqlite:///{tmp_path / 'history.db'}"
    monkeypatch.setattr(history, "_engines", {})
    # JSONB is Postgres-only; the column type is swapped for the SQLite run.
    monkeypatch.setattr(
        history.query_history.c.sources, "type", sa.JSON(), raising=False
    )
    monkeypatch.setenv("DATABASE_URL", url)
    return url


def test_record_then_read_back(db):
    entry_id = history.record(
        "who said it?", "Ruby did.", ["http://example.com/a"], latency_ms=12.5, model="m"
    )

    entry = history.get(entry_id)
    assert entry["query"] == "who said it?"
    assert entry["answer"] == "Ruby did."
    assert entry["sources"] == ["http://example.com/a"]
    assert entry["refused"] is False
    assert entry["latency_ms"] == 12.5
    assert entry["model"] == "m"


def test_an_answer_without_sources_is_marked_refused(db):
    entry_id = history.record("off topic?", "I don't have enough context.", [])

    assert history.get(entry_id)["refused"] is True


def test_recent_returns_newest_first_and_respects_limit(db):
    for i in range(4):
        history.record(f"q{i}", "a", [])

    recent = history.recent(limit=2)

    assert len(recent) == 2
    assert [e["query"] for e in recent] == ["q3", "q2"]


def test_disabled_history_records_nothing(db, monkeypatch):
    monkeypatch.setenv("RAG_HISTORY", "false")

    assert history.record("q", "a", []) is None
    assert history.recent() == []


def test_a_write_failure_does_not_raise(db, monkeypatch, caplog):
    """The whole point of the try/except: a broken database must not lose the answer."""
    monkeypatch.setattr(
        history, "_engine", lambda connection=None: (_ for _ in ()).throw(OSError("down"))
    )

    assert history.record("q", "a", []) is None


def test_delete_all_clears_the_table(db):
    history.record("q1", "a", [])
    history.record("q2", "a", [])

    assert history.delete_all() == 2
    assert history.recent() == []


def test_get_returns_none_for_an_unknown_id(db):
    assert history.get("no-such-id") is None
