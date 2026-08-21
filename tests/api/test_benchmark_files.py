import io
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_upload_list_and_delete_test_file(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    monkeypatch.setattr("src.benchmark.files.UPLOAD_DIR", upload_dir)
    monkeypatch.setattr("src.benchmark.files.MANIFEST_PATH", upload_dir / "manifest.json")

    # Initial list is empty
    resp = client.get("/api/benchmark/test-files")
    assert resp.status_code == 200
    assert resp.json() == []

    # Upload test file
    csv_content = b"question,answer\nwhat is rag?,retrieval augmented generation\n"
    upload_resp = client.post(
        "/api/benchmark/test-files/upload",
        files={"file": ("custom_eval.csv", io.BytesIO(csv_content), "text/csv")},
    )
    assert upload_resp.status_code == 201
    file_id = upload_resp.json()["id"]

    # List files now has 1
    resp = client.get("/api/benchmark/test-files")
    assert resp.status_code == 200
    files = resp.json()
    assert len(files) == 1
    assert files[0]["name"] == "custom_eval.csv"

    # Delete uploaded file
    del_resp = client.delete(f"/api/benchmark/test-files/{file_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["id"] == file_id

    # List files is empty again
    resp_after = client.get("/api/benchmark/test-files")
    assert resp_after.status_code == 200
    assert len(resp_after.json()) == 0

    # Delete nonexistent file returns 404
    del_nonexistent = client.delete("/api/benchmark/test-files/nonexistent_id")
    assert del_nonexistent.status_code == 404
