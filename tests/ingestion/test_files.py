"""Ingested-file provenance, exercised against SQLite rather than Postgres.

Mirrors tests/rag/test_history.py -- same throwaway-database approach, since
the table is created from the same SQLAlchemy metadata either way.
"""

import io

import pytest
import sqlalchemy as sa

from src.db import engine as db_engine
from src.ingestion import files


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A throwaway database, with the module's engine cache cleared around it."""
    url = f"sqlite:///{tmp_path / 'files.db'}"
    monkeypatch.setattr(db_engine, "_engines", {})
    monkeypatch.setattr(db_engine, "_initialized", set())
    # JSONB is Postgres-only; the column type is swapped for the SQLite run.
    monkeypatch.setattr(
        files.ingested_files.c.chunk_ids, "type", sa.JSON(), raising=False
    )
    monkeypatch.setenv("DATABASE_URL", url)
    return url


def test_hash_file_is_stable_for_the_same_bytes():
    assert files.hash_file(io.BytesIO(b"hello world")) == files.hash_file(
        io.BytesIO(b"hello world")
    )


def test_hash_file_differs_for_different_bytes():
    assert files.hash_file(io.BytesIO(b"a")) != files.hash_file(io.BytesIO(b"b"))


def test_record_then_find_by_hash(db):
    files.record("mine.csv", "data/uploads/mine.csv", "abc123", size_bytes=11, documents=3)

    entry = files.find_by_hash("abc123")

    assert entry["filename"] == "mine.csv"
    assert entry["stored_path"] == "data/uploads/mine.csv"
    assert entry["size_bytes"] == 11
    assert entry["documents"] == 3


def test_find_by_hash_returns_none_for_unknown_hash(db):
    assert files.find_by_hash("nope") is None


def test_recent_returns_newest_first_and_respects_limit(db):
    for i in range(4):
        files.record(f"f{i}.csv", f"data/uploads/f{i}.csv", f"hash{i}", size_bytes=1)

    recent = files.recent(limit=2)

    assert len(recent) == 2
    assert [e["filename"] for e in recent] == ["f3.csv", "f2.csv"]


def test_get_returns_the_record_by_id(db):
    entry_id = files.record("mine.csv", "data/uploads/mine.csv", "abc123", size_bytes=11)

    entry = files.get(entry_id)

    assert entry["filename"] == "mine.csv"
    assert entry["chunk_ids"] is None


def test_get_returns_none_for_an_unknown_id(db):
    assert files.get("no-such-id") is None


def test_set_chunk_ids_fills_them_in_after_the_fact(db):
    entry_id = files.record("mine.csv", "data/uploads/mine.csv", "abc123", size_bytes=11)

    files.set_chunk_ids(entry_id, ["c1", "c2"])

    assert files.get(entry_id)["chunk_ids"] == ["c1", "c2"]


def test_delete_removes_the_record(db):
    entry_id = files.record("mine.csv", "data/uploads/mine.csv", "abc123", size_bytes=11)

    assert files.delete(entry_id) is True
    assert files.get(entry_id) is None


def test_delete_returns_false_for_an_unknown_id(db):
    assert files.delete("no-such-id") is False


def test_a_write_failure_does_not_raise(db, monkeypatch):
    """Same failure policy as query history: never break a caller over provenance."""
    monkeypatch.setattr(
        files, "_engine", lambda connection=None: (_ for _ in ()).throw(OSError("down"))
    )

    assert files.record("f.csv", "data/uploads/f.csv", "hash", size_bytes=1) is None
