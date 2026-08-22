import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.routes import documents as documents_route
from src.api.routes import ingest as ingest_route

CSV = "passage,id\nfirst passage,9797\nsecond passage,11906\n"


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    directory = tmp_path / "uploads"
    directory.mkdir()
    monkeypatch.setattr(ingest_route, "UPLOAD_DIR", directory)
    return directory


@pytest.fixture
def store(monkeypatch):
    entries: dict[str, dict] = {}
    monkeypatch.setattr(
        documents_route.file_history, "get", lambda file_id: entries.get(file_id)
    )
    return entries


@pytest.fixture
def client():
    return TestClient(create_app())


def _add(store, uploads, name="corpus.csv", content=CSV, units=2, path=None):
    stored = uploads / name
    if content is not None:
        stored.write_text(content, encoding="utf-8")
    store["f1"] = {
        "id": "f1",
        "created_at": "2026-08-19T12:00:00+00:00",
        "filename": name,
        "stored_path": str(path if path is not None else stored),
        "sha256": "abc",
        "size_bytes": len(content or ""),
        "documents": units,
        "chunk_ids": ["c1", "c2"],
    }
    return stored


def test_meta_describes_the_document(client, store, uploads):
    _add(store, uploads)

    body = client.get("/api/documents/f1").json()

    assert body["filename"] == "corpus.csv"
    assert body["extension"] == ".csv"
    assert body["unit_kind"] == "row"
    assert body["units"] == 2
    assert body["chunks"] == 2
    assert body["columns"] == ["passage", "id"]


def test_meta_reports_columns_only_for_tabular_documents(client, store, uploads):
    _add(store, uploads, name="notes.txt", content="hello", units=1)

    body = client.get("/api/documents/f1").json()

    assert body["unit_kind"] == "line"
    assert body["columns"] is None


def test_units_returns_a_window(client, store, uploads):
    _add(store, uploads)

    units = client.get("/api/documents/f1/units").json()

    assert [u["index"] for u in units] == [1, 2]
    assert units[0]["text"] == "first passage"
    assert units[0]["label"] == "row 1"
    assert units[0]["key"] == "9797"


def test_units_respects_offset_and_limit(client, store, uploads):
    _add(store, uploads)

    units = client.get("/api/documents/f1/units?offset=1&limit=1").json()

    assert [u["index"] for u in units] == [2]


def test_units_rejects_an_oversized_page(client, store, uploads):
    _add(store, uploads)

    response = client.get(f"/api/documents/f1/units?limit={documents_route.MAX_PAGE + 1}")

    assert response.status_code == 422


def test_single_unit_is_addressed_the_way_a_citation_addresses_it(client, store, uploads):
    _add(store, uploads)

    body = client.get("/api/documents/f1/units/2").json()

    assert body["index"] == 2
    assert body["text"] == "second passage"


def test_a_missing_unit_is_a_404(client, store, uploads):
    _add(store, uploads)

    response = client.get("/api/documents/f1/units/99")

    assert response.status_code == 404
    assert "row 99" in response.json()["detail"]


def test_unknown_document_is_a_404(client, store, uploads):
    assert client.get("/api/documents/nope").status_code == 404


def test_a_record_pointing_outside_the_upload_directory_is_refused(
    client, store, uploads, tmp_path
):
    secret = tmp_path / "secret.csv"
    secret.write_text("passage\ntop secret\n", encoding="utf-8")
    _add(store, uploads, path=secret)

    response = client.get("/api/documents/f1")

    assert response.status_code == 403


def test_a_traversal_path_is_refused(client, store, uploads, tmp_path):
    secret = tmp_path / "secret.csv"
    secret.write_text("passage\ntop secret\n", encoding="utf-8")
    _add(store, uploads, path=uploads / ".." / "secret.csv")

    response = client.get("/api/documents/f1/units")

    assert response.status_code == 403
    assert "top secret" not in response.text


def test_a_deleted_stored_copy_reports_gone(client, store, uploads):
    stored = _add(store, uploads)
    stored.unlink()

    response = client.get("/api/documents/f1")

    assert response.status_code == 410


def test_an_unreadable_document_is_reported_not_crashed(client, store, uploads):
    _add(store, uploads, name="broken.jsonl", content="{not json\n", units=1)

    response = client.get("/api/documents/f1/units")

    assert response.status_code == 422
