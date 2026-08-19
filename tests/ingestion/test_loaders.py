import csv

from src.ingestion.loaders import load_documents


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "source_url", "text"])
        writer.writeheader()
        writer.writerows(rows)


def test_load_documents_maps_fields_to_document(tmp_path):
    csv_path = tmp_path / "docs.csv"
    _write_csv(
        csv_path,
        [{"index": "1", "source_url": "http://example.com/a", "text": "hello world"}],
    )

    docs = load_documents(csv_path)

    assert len(docs) == 1
    assert docs[0].page_content == "hello world"
    assert docs[0].metadata == {"source": "http://example.com/a", "row_index": "1"}


def test_load_documents_skips_empty_text_rows(tmp_path):
    csv_path = tmp_path / "docs.csv"
    _write_csv(
        csv_path,
        [
            {"index": "1", "source_url": "http://example.com/a", "text": ""},
            {"index": "2", "source_url": "http://example.com/b", "text": "   "},
            {"index": "3", "source_url": "http://example.com/c", "text": "kept"},
        ],
    )

    docs = load_documents(csv_path)

    assert len(docs) == 1
    assert docs[0].page_content == "kept"


def test_load_documents_empty_file_returns_empty_list(tmp_path):
    csv_path = tmp_path / "docs.csv"
    _write_csv(csv_path, [])

    assert load_documents(csv_path) == []
