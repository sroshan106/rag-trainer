import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import schemas
from src.api.routes import ingest as ingest_route
from src.api.app import create_app
from src.jobs.runner import JobRunner

CSV = "index,source_url,text\n1,http://example.com,hello world\n"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest_route, "runner", JobRunner())
    monkeypatch.setattr(ingest_route, "UPLOAD_DIR", tmp_path / "uploads")
    return TestClient(create_app())


@pytest.fixture
def file_store(monkeypatch):
    entries: dict[str, dict] = {}

    def find_by_hash(sha256):
        return entries.get(sha256)

    def record(
        filename,
        stored_path,
        sha256,
        size_bytes,
        documents=None,
        index_columns=None,
        **kwargs,
    ):
        entries[sha256] = {
            "id": sha256,
            "created_at": "2026-08-19T12:00:00+00:00",
            "filename": filename,
            "stored_path": str(stored_path),
            "sha256": sha256,
            "size_bytes": size_bytes,
            "documents": documents,
            "chunk_ids": None,
            "index_columns": index_columns,
        }
        return sha256

    def recent(limit=50):
        return list(entries.values())[:limit]

    def get(entry_id):
        return next((e for e in entries.values() if e["id"] == entry_id), None)

    def set_chunk_ids(entry_id, chunk_ids):
        entry = get(entry_id)
        if entry is not None:
            entry["chunk_ids"] = list(chunk_ids)

    def delete(entry_id):
        for sha, entry in list(entries.items()):
            if entry["id"] == entry_id:
                del entries[sha]
                return True
        return False

    monkeypatch.setattr(ingest_route.file_history, "find_by_hash", find_by_hash)
    monkeypatch.setattr(ingest_route.file_history, "record", record)
    monkeypatch.setattr(ingest_route.file_history, "recent", recent)
    monkeypatch.setattr(ingest_route.file_history, "get", get)
    monkeypatch.setattr(ingest_route.file_history, "set_chunk_ids", set_chunk_ids)
    monkeypatch.setattr(ingest_route.file_history, "delete", delete)
    return entries


@pytest.fixture
def stub_delete_chunks(monkeypatch):
    calls = []
    monkeypatch.setattr(ingest_route, "delete_chunks", lambda ids: calls.append(list(ids)))
    return calls


@pytest.fixture
def stub_ingest(monkeypatch):
    calls = []

    def fake(path, progress=None, splitter=None, file_id=None, filename=None):
        calls.append(
            {
                "path": str(path),
                "splitter": splitter,
                "file_id": file_id,
                "filename": filename,
            }
        )
        if progress:
            progress(1.0, "done")
        return {
            "path": str(path),
            "documents": 1,
            "chunks": 1,
            "chunk_ids": ["chunk-1"],
            "splitter": splitter,
            "file_id": file_id,
            "filename": filename,
            "index_built": False,
        }

    monkeypatch.setattr(ingest_route, "ingest", fake)
    return calls


def _wait_idle(timeout=5.0):
    deadline = threading.Event()
    deadline.wait(0)
    for _ in range(int(timeout * 100)):
        if ingest_route.runner.active(ingest_route.JOB_KIND) is None:
            return
        deadline.wait(0.01)
    raise AssertionError("ingest job never finished")


def test_upload_ingests_the_uploaded_file(client, stub_ingest, file_store, tmp_path):
    response = client.post(
        "/api/ingest/upload", files={"file": ("mine.csv", CSV, "text/csv")}
    )

    assert response.status_code == 202
    _wait_idle()
    assert len(stub_ingest) == 1
    assert stub_ingest[0]["path"].endswith("mine.csv")


def test_upload_records_the_file_and_its_hash(client, stub_ingest, file_store):
    client.post("/api/ingest/upload", files={"file": ("mine.csv", CSV, "text/csv")})
    _wait_idle()

    assert len(file_store) == 1
    entry = next(iter(file_store.values()))
    assert entry["filename"] == "mine.csv"
    assert len(entry["sha256"]) == 64
    assert entry["documents"] == 1


def test_second_upload_of_identical_bytes_is_refused(client, stub_ingest, file_store):
    first = client.post("/api/ingest/upload", files={"file": ("mine.csv", CSV, "text/csv")})
    assert first.status_code == 202
    _wait_idle()

    second = client.post("/api/ingest/upload", files={"file": ("again.csv", CSV, "text/csv")})

    assert second.status_code == 409
    assert "mine.csv" in second.json()["detail"]
    assert len(stub_ingest) == 1
    assert len(file_store) == 1


def test_duplicate_upload_does_not_leave_a_second_copy_on_disk(client, stub_ingest, file_store, tmp_path):
    client.post("/api/ingest/upload", files={"file": ("mine.csv", CSV, "text/csv")})
    _wait_idle()

    client.post("/api/ingest/upload", files={"file": ("again.csv", CSV, "text/csv")})

    assert len(list(ingest_route.UPLOAD_DIR.glob("*"))) == 1


def test_upload_accepts_a_csv_with_arbitrary_columns(client, stub_ingest, file_store):
    response = client.post(
        "/api/ingest/upload",
        files={"file": ("bioasq.csv", "passage,id\nsome text,9797\n", "text/csv")},
    )

    assert response.status_code == 202
    _wait_idle()
    assert len(stub_ingest) == 1


def test_upload_passes_the_original_filename_through_for_citations(
    client, stub_ingest, file_store
):
    client.post("/api/ingest/upload", files={"file": ("mine.csv", CSV, "text/csv")})
    _wait_idle()

    assert stub_ingest[0]["filename"] == "mine.csv"
    assert stub_ingest[0]["file_id"] is not None
    assert "mine.csv" not in Path(stub_ingest[0]["path"]).name.split("-")[0]


def test_upload_rejects_unsupported_extension(client, stub_ingest, file_store):
    response = client.post(
        "/api/ingest/upload", files={"file": ("notes.docx", "hello", "text/plain")}
    )

    assert response.status_code == 415
    assert stub_ingest == []


def test_upload_rejects_a_csv_with_no_usable_rows(client, stub_ingest, file_store):
    response = client.post(
        "/api/ingest/upload", files={"file": ("empty.csv", "text\n\n", "text/csv")}
    )

    assert response.status_code == 422
    assert stub_ingest == []


def test_uploaded_filename_cannot_escape_the_upload_directory(client, stub_ingest, file_store):
    response = client.post(
        "/api/ingest/upload", files={"file": ("../../evil.csv", CSV, "text/csv")}
    )

    assert response.status_code == 202
    _wait_idle()
    stored = Path(stub_ingest[0]["path"]).resolve()
    assert stored.parent == ingest_route.UPLOAD_DIR.resolve()
    assert stored.exists()


def test_history_lists_recorded_files(client, file_store):
    file_store["abc123"] = {
        "id": "1",
        "created_at": "2026-08-19T12:00:00+00:00",
        "filename": "mine.csv",
        "stored_path": "data/uploads/mine.csv",
        "sha256": "abc123",
        "size_bytes": 42,
        "documents": 3,
    }

    response = client.get("/api/ingest/history")

    assert response.status_code == 200
    assert response.json()[0]["filename"] == "mine.csv"


def test_upload_records_chunk_ids_once_the_job_finishes(client, stub_ingest, file_store):
    client.post("/api/ingest/upload", files={"file": ("mine.csv", CSV, "text/csv")})
    _wait_idle()

    entry = next(iter(file_store.values()))
    assert entry["chunk_ids"] == ["chunk-1"]


def test_upload_passes_the_chosen_splitter_through(client, stub_ingest, file_store):
    client.post(
        "/api/ingest/upload",
        files={"file": ("mine.csv", CSV, "text/csv")},
        data={"splitter": "token"},
    )
    _wait_idle()

    entry = next(iter(file_store.values()))
    assert entry["chunk_ids"] == ["chunk-1"]


def test_upload_rejects_an_unknown_splitter(client, stub_ingest, file_store):
    response = client.post(
        "/api/ingest/upload",
        files={"file": ("mine.csv", CSV, "text/csv")},
        data={"splitter": "nope"},
    )

    assert response.status_code == 422
    assert stub_ingest == []


def test_list_splitters_reports_available_and_default(client):
    response = client.get("/api/ingest/splitters")

    assert response.status_code == 200
    body = response.json()
    assert "recursive" in body["splitters"]
    assert body["default"] == "recursive"


def test_delete_removes_vectors_copy_and_record(client, stub_ingest, file_store, stub_delete_chunks):
    client.post("/api/ingest/upload", files={"file": ("mine.csv", CSV, "text/csv")})
    _wait_idle()
    entry = next(iter(file_store.values()))
    stored = Path(entry["stored_path"])
    assert stored.exists()

    response = client.delete(f"/api/ingest/files/{entry['id']}")

    assert response.status_code == 200
    assert response.json() == {"deleted_chunks": 1, "filename": "mine.csv"}
    assert stub_delete_chunks == [["chunk-1"]]
    assert not stored.exists()
    assert file_store == {}


def test_delete_unknown_file_is_a_404(client, file_store, stub_delete_chunks):
    response = client.delete("/api/ingest/files/nope")

    assert response.status_code == 404
    assert stub_delete_chunks == []


def test_delete_refused_while_an_ingest_is_running(client, monkeypatch, file_store, stub_delete_chunks):
    release = threading.Event()
    monkeypatch.setattr(
        ingest_route,
        "ingest",
        lambda path, progress=None, splitter=None, file_id=None, filename=None: (
            release.wait(5),
            {"path": path},
        )[1],
    )
    file_store["h"] = {
        "id": "1",
        "created_at": "2026-08-19T12:00:00+00:00",
        "filename": "mine.csv",
        "stored_path": "data/uploads/mine.csv",
        "sha256": "h",
        "size_bytes": 1,
        "documents": 1,
        "chunk_ids": ["c1"],
    }
    client.post("/api/ingest/upload", files={"file": ("other.csv", CSV, "text/csv")})

    response = client.delete("/api/ingest/files/1")

    assert response.status_code == 409
    assert stub_delete_chunks == []

    release.set()
    _wait_idle()


def test_second_upload_is_refused_while_one_runs(client, monkeypatch, file_store):
    release = threading.Event()

    def blocking(path, progress=None, splitter=None, file_id=None, filename=None):
        release.wait(5)
        return {"path": path, "documents": 0, "chunks": 0, "index_built": False}

    monkeypatch.setattr(ingest_route, "ingest", blocking)

    first = client.post("/api/ingest/upload", files={"file": ("mine.csv", CSV, "text/csv")})
    assert first.status_code == 202

    second = client.post("/api/ingest/upload", files={"file": ("other.csv", CSV, "text/csv")})
    assert second.status_code == 409

    release.set()
    _wait_idle()


def test_active_reports_the_running_job_then_nothing(client, monkeypatch, file_store):
    release = threading.Event()
    monkeypatch.setattr(
        ingest_route,
        "ingest",
        lambda path, progress=None, splitter=None, file_id=None, filename=None: (
            release.wait(5),
            {"path": path},
        )[1],
    )

    started = client.post(
        "/api/ingest/upload", files={"file": ("mine.csv", CSV, "text/csv")}
    ).json()
    assert client.get("/api/ingest/active").json()["id"] == started["id"]

    release.set()
    _wait_idle()
    assert client.get("/api/ingest/active").json() is None


def test_benchmark_default_workers_matches_the_measured_value():
    from tests.benchmark.run_benchmark import DEFAULT_WORKERS

    assert schemas.BenchmarkRequest().workers == DEFAULT_WORKERS


def test_upload_with_index_columns(client, monkeypatch, file_store):
    called_with = {}

    def fake_ingest(path, progress=None, splitter=None, file_id=None, filename=None, index_columns=None):
        called_with["index_columns"] = index_columns
        return {"path": path, "documents": 1, "chunks": 1, "chunk_ids": ["c1"], "index_built": True}

    monkeypatch.setattr(ingest_route, "ingest", fake_ingest)

    csv_data = "col1,col2,col3\nval1,val2,val3\n"
    resp = client.post(
        "/api/ingest/upload",
        files={"file": ("custom.csv", csv_data, "text/csv")},
        data={"index_columns": '["col1", "col3"]'},
    )
    assert resp.status_code == 202
    _wait_idle()

    assert called_with["index_columns"] == ["col1", "col3"]
    record = client.get("/api/ingest/history").json()[0]
    assert record["index_columns"] == ["col1", "col3"]
