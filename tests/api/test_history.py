"""History routes, with the store stubbed -- the store's own tests cover storage."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.routes import history as history_route

ENTRY = {
    "id": "abc",
    "created_at": "2026-08-19T12:00:00+00:00",
    "query": "who said it?",
    "answer": "Ruby did.",
    "sources": ["http://example.com/a"],
    "refused": False,
    "latency_ms": 12.5,
    "model": "llama3.2:3b",
}


@pytest.fixture
def client():
    return TestClient(create_app())


def test_list_returns_entries(client, monkeypatch):
    monkeypatch.setattr(history_route.history, "recent", lambda limit: [ENTRY])

    body = client.get("/api/history").json()

    assert body[0]["query"] == "who said it?"
    assert body[0]["sources"] == ["http://example.com/a"]


def test_limit_is_passed_through_and_bounded(client, monkeypatch):
    seen = []
    monkeypatch.setattr(
        history_route.history, "recent", lambda limit: seen.append(limit) or []
    )

    client.get("/api/history?limit=7")
    assert seen == [7]

    assert client.get("/api/history?limit=0").status_code == 422
    assert client.get("/api/history?limit=99999").status_code == 422


def test_unknown_entry_is_a_404(client, monkeypatch):
    monkeypatch.setattr(history_route.history, "get", lambda entry_id: None)

    assert client.get("/api/history/nope").status_code == 404


def test_delete_reports_how_many_rows_went(client, monkeypatch):
    monkeypatch.setattr(history_route.history, "delete_all", lambda: 3)

    assert client.delete("/api/history").json() == {"deleted": 3}


def test_delete_one_entry_reports_the_row(client, monkeypatch):
    seen = []
    monkeypatch.setattr(
        history_route.history, "delete", lambda entry_id: seen.append(entry_id) or 1
    )

    resp = client.delete("/api/history/abc")

    assert resp.status_code == 200
    assert resp.json() == {"deleted": 1}
    assert seen == ["abc"]


def test_deleting_an_unknown_entry_is_a_404(client, monkeypatch):
    monkeypatch.setattr(history_route.history, "delete", lambda entry_id: 0)

    resp = client.delete("/api/history/nope")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "no such history entry"


def test_status_is_part_of_the_response(client, monkeypatch):
    monkeypatch.setattr(
        history_route.history, "recent", lambda limit: [{**ENTRY, "status": "cancelled"}]
    )

    assert client.get("/api/history").json()[0]["status"] == "cancelled"
