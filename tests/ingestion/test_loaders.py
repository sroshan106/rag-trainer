import json

import pytest

from src.ingestion.loaders import UnsupportedFileType, load_documents


def test_load_documents_stamps_provenance_onto_every_document(tmp_path):
    path = tmp_path / "docs.csv"
    path.write_text("passage,id\nhello world,9797\n", encoding="utf-8")

    docs = load_documents(path, file_id="file-1", filename="original.csv")

    assert len(docs) == 1
    assert docs[0].metadata["file_id"] == "file-1"
    assert docs[0].metadata["filename"] == "original.csv"
    assert docs[0].metadata["unit_kind"] == "row"
    assert docs[0].metadata["unit_index"] == 1


def test_unit_index_locates_the_row_while_row_index_holds_the_dataset_key(tmp_path):
    path = tmp_path / "corpus.csv"
    path.write_text("passage,id\nfirst,9797\nsecond,15908939\n", encoding="utf-8")

    docs = load_documents(path)

    assert [d.metadata["unit_index"] for d in docs] == [1, 2]
    assert [d.metadata["row_index"] for d in docs] == ["9797", "15908939"]


def test_row_index_is_none_when_the_file_declares_no_identifier(tmp_path):
    path = tmp_path / "plain.csv"
    path.write_text("passage\njust text\n", encoding="utf-8")

    assert load_documents(path)[0].metadata["row_index"] is None


def test_filename_defaults_to_the_path_when_not_given(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")

    assert load_documents(path)[0].metadata["filename"] == "notes.txt"


def test_missing_file_id_is_tolerated(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")

    docs = load_documents(path, file_id=None)

    assert docs[0].metadata["file_id"] is None


def test_a_url_column_rides_along_when_present(tmp_path):
    path = tmp_path / "docs.csv"
    path.write_text("text,source_url\nhello,https://example.com/a\n", encoding="utf-8")

    assert load_documents(path)[0].metadata["url"] == "https://example.com/a"


def test_whole_row_is_embedded_without_picking_a_text_column(tmp_path):
    path = tmp_path / "bioasq.csv"
    path.write_text("passage,id\nsome biomedical text,9797\n", encoding="utf-8")

    content = load_documents(path)[0].page_content

    assert "some biomedical text" in content


def test_load_documents_reads_every_supported_format(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "a.md").write_text("# hi", encoding="utf-8")
    (tmp_path / "a.json").write_text(json.dumps([{"text": "x"}]), encoding="utf-8")
    (tmp_path / "a.jsonl").write_text('{"text": "y"}\n', encoding="utf-8")

    for name in ("a.txt", "a.md", "a.json", "a.jsonl"):
        assert len(load_documents(tmp_path / name)) == 1, name


def test_load_documents_raises_for_unsupported_extension(tmp_path):
    path = tmp_path / "notes.docx"
    path.write_text("hello", encoding="utf-8")

    with pytest.raises(UnsupportedFileType):
        load_documents(path)
