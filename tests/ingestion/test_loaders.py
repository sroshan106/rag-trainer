import csv
import json

import pytest

from src.ingestion.loaders import UnsupportedFileType, load_documents


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


def test_load_documents_reads_a_txt_file(tmp_path):
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("hello world", encoding="utf-8")

    docs = load_documents(txt_path)

    assert len(docs) == 1
    assert docs[0].page_content == "hello world"


def test_load_documents_reads_a_md_file(tmp_path):
    md_path = tmp_path / "notes.md"
    md_path.write_text("# heading\nbody text", encoding="utf-8")

    docs = load_documents(md_path)

    assert len(docs) == 1
    assert "body text" in docs[0].page_content


def test_load_documents_empty_txt_returns_empty_list(tmp_path):
    txt_path = tmp_path / "empty.txt"
    txt_path.write_text("   ", encoding="utf-8")

    assert load_documents(txt_path) == []


def test_load_documents_reads_a_json_array(tmp_path):
    json_path = tmp_path / "docs.json"
    json_path.write_text(
        json.dumps([{"text": "hello", "source_url": "http://a"}, {"text": "  "}]),
        encoding="utf-8",
    )

    docs = load_documents(json_path)

    assert len(docs) == 1
    assert docs[0].page_content == "hello"
    assert docs[0].metadata["source"] == "http://a"


def test_load_documents_reads_a_jsonl_file(tmp_path):
    jsonl_path = tmp_path / "docs.jsonl"
    jsonl_path.write_text('{"text": "a"}\n{"text": "b"}\n', encoding="utf-8")

    docs = load_documents(jsonl_path)

    assert [d.page_content for d in docs] == ["a", "b"]


def test_load_documents_maps_a_passage_column_by_alias(tmp_path):
    csv_path = tmp_path / "bioasq.csv"
    csv_path.write_text(
        'passage,id\n"a fairly long passage of biomedical text here",9797\n',
        encoding="utf-8",
    )

    docs = load_documents(csv_path)

    assert len(docs) == 1
    assert docs[0].page_content == "a fairly long passage of biomedical text here"
    assert docs[0].metadata["row_index"] == "9797"


def test_load_documents_guesses_the_text_column_by_content(tmp_path):
    csv_path = tmp_path / "unknown_headers.csv"
    csv_path.write_text(
        "ref,notes\n"
        '101,"this column just happens to hold the actual long free text content"\n',
        encoding="utf-8",
    )

    docs = load_documents(csv_path)

    assert len(docs) == 1
    assert "long free text content" in docs[0].page_content


def test_load_documents_raises_when_no_column_looks_like_text(tmp_path):
    csv_path = tmp_path / "ids_only.csv"
    csv_path.write_text("index,source_url\n1,http://x\n", encoding="utf-8")

    with pytest.raises(Exception, match="text"):
        load_documents(csv_path)


def test_load_documents_raises_for_unsupported_extension(tmp_path):
    docx_path = tmp_path / "notes.docx"
    docx_path.write_text("hello", encoding="utf-8")

    with pytest.raises(UnsupportedFileType):
        load_documents(docx_path)
