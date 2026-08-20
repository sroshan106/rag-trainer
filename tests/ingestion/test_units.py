"""Unit boundaries and numbering.

These are the contract the document viewer and the citations both depend on:
if a unit index moves, every stored citation silently starts pointing at the
wrong text, so the numbering is pinned here rather than left implicit.
"""

import json

import pytest

from src.ingestion.units import (
    KIND_LINE,
    KIND_PAGE,
    KIND_ROW,
    UnreadableFile,
    UnsupportedFileType,
    csv_columns,
    is_supported,
    iter_units,
    unit_kind,
)


def test_csv_rows_are_numbered_from_one_excluding_the_header(tmp_path):
    path = tmp_path / "docs.csv"
    path.write_text("a,b\nfirst,x\nsecond,y\n", encoding="utf-8")

    units = list(iter_units(path))

    assert [u.index for u in units] == [1, 2]
    assert units[0].kind == KIND_ROW


def test_csv_row_serialises_every_content_column(tmp_path):
    path = tmp_path / "docs.csv"
    path.write_text("question,answer\nwhat?,because\n", encoding="utf-8")

    text = list(iter_units(path))[0].text

    assert "question: what?" in text
    assert "answer: because" in text


def test_a_single_content_column_alongside_an_id_emits_bare_text(tmp_path):
    """rag-mini-bioasq's shape: once id is lifted out, only the passage is left."""
    path = tmp_path / "docs.csv"
    path.write_text("passage,id\nsome text here,9797\n", encoding="utf-8")

    assert list(iter_units(path))[0].text == "some text here"


def test_single_column_csv_emits_the_bare_value(tmp_path):
    path = tmp_path / "docs.csv"
    path.write_text("passage\njust the text\n", encoding="utf-8")

    assert list(iter_units(path))[0].text == "just the text"


def test_csv_skips_empty_columns_but_keeps_the_row_number(tmp_path):
    path = tmp_path / "docs.csv"
    path.write_text("a,b\n,kept\n", encoding="utf-8")

    units = list(iter_units(path))

    assert units[0].text == "b: kept"
    assert units[0].index == 1


def test_csv_row_with_no_content_yields_no_unit_but_does_not_shift_later_rows(tmp_path):
    path = tmp_path / "docs.csv"
    path.write_text("a,b\n,\nkept,also\n", encoding="utf-8")

    units = list(iter_units(path))

    assert len(units) == 1
    # Still row 2 -- numbering follows the file, not the surviving units.
    assert units[0].index == 2


def test_columns_lifted_into_metadata_are_not_also_embedded(tmp_path):
    """Shaped after the benchmark corpus, whose source_url column is huge.

    Embedding the id and the link would spend the chunk's budget on text no
    question can be asked about, and would only appear in a long row's first
    chunk anyway.
    """
    path = tmp_path / "documents.csv"
    path.write_text(
        "index,source_url,text\n0,https://example.com/very/long/link,the real content\n",
        encoding="utf-8",
    )

    unit = list(iter_units(path))[0]

    assert unit.text == "the real content"
    assert unit.key == "0"
    assert unit.url == "https://example.com/very/long/link"


def test_a_row_of_only_metadata_yields_no_unit(tmp_path):
    """Nothing left to embed once the id and link are lifted out."""
    path = tmp_path / "bare.csv"
    path.write_text("index,source_url\n0,https://example.com/a\n", encoding="utf-8")

    assert list(iter_units(path)) == []


def test_csv_key_comes_from_an_id_column_and_is_not_the_position(tmp_path):
    """The distinction the benchmark's recall scoring depends on.

    Shaped after rag-mini-bioasq, where the passage with id 31776899 is the
    37946th row -- position and dataset key are unrelated numbers.
    """
    path = tmp_path / "corpus.csv"
    path.write_text("passage,id\nfirst,9797\nsecond,15908939\n", encoding="utf-8")

    units = list(iter_units(path))

    assert [u.index for u in units] == [1, 2]
    assert [u.key for u in units] == ["9797", "15908939"]


def test_csv_key_prefers_an_explicit_index_column(tmp_path):
    """The built-in benchmark corpus numbers its index column from 0."""
    path = tmp_path / "documents.csv"
    path.write_text("index,source_url,text\n0,https://x/a,hello\n", encoding="utf-8")

    unit = list(iter_units(path))[0]

    assert unit.index == 1
    assert unit.key == "0"


def test_csv_key_is_none_when_no_identifier_column_exists(tmp_path):
    path = tmp_path / "plain.csv"
    path.write_text("passage\njust text\n", encoding="utf-8")

    assert list(iter_units(path))[0].key is None


def test_json_records_carry_their_key(tmp_path):
    path = tmp_path / "docs.json"
    path.write_text(json.dumps([{"text": "a", "doc_id": "x1"}]), encoding="utf-8")

    assert list(iter_units(path))[0].key == "x1"


def test_csv_picks_up_a_url_column_when_one_exists(tmp_path):
    path = tmp_path / "docs.csv"
    path.write_text("text,source_url\nhello,https://example.com/a\n", encoding="utf-8")

    assert list(iter_units(path))[0].url == "https://example.com/a"


def test_csv_ignores_a_url_column_that_is_not_a_link(tmp_path):
    path = tmp_path / "docs.csv"
    path.write_text("text,url\nhello,not-a-link\n", encoding="utf-8")

    assert list(iter_units(path))[0].url is None


def test_csv_with_no_rows_yields_nothing(tmp_path):
    path = tmp_path / "docs.csv"
    path.write_text("a,b\n", encoding="utf-8")

    assert list(iter_units(path)) == []


def test_empty_csv_yields_nothing(tmp_path):
    path = tmp_path / "docs.csv"
    path.write_text("", encoding="utf-8")

    assert list(iter_units(path)) == []


def test_json_array_is_numbered_by_position(tmp_path):
    path = tmp_path / "docs.json"
    path.write_text(json.dumps([{"text": "a"}, {"text": "b"}]), encoding="utf-8")

    units = list(iter_units(path))

    assert [u.index for u in units] == [1, 2]
    assert units[0].kind == KIND_ROW


def test_json_accepts_plain_strings(tmp_path):
    path = tmp_path / "docs.json"
    path.write_text(json.dumps(["hello", "world"]), encoding="utf-8")

    assert [u.text for u in iter_units(path)] == ["hello", "world"]


def test_json_accepts_a_bare_object(tmp_path):
    path = tmp_path / "docs.json"
    path.write_text(json.dumps({"text": "only one"}), encoding="utf-8")

    assert [u.text for u in iter_units(path)] == ["only one"]


def test_invalid_json_is_reported_as_unreadable(tmp_path):
    path = tmp_path / "docs.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(UnreadableFile):
        list(iter_units(path))


def test_jsonl_is_numbered_by_physical_line(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text('{"text": "a"}\n\n{"text": "b"}\n', encoding="utf-8")

    units = list(iter_units(path))

    # Blank second line yields no unit, and the third line keeps its number.
    assert [u.index for u in units] == [1, 3]
    assert units[0].kind == KIND_LINE


def test_invalid_jsonl_names_the_offending_line(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text('{"text": "a"}\nnot json\n', encoding="utf-8")

    with pytest.raises(UnreadableFile, match="line 2"):
        list(iter_units(path))


def test_text_splits_into_paragraphs_indexed_by_starting_line(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("first para\nstill first\n\nsecond para\n", encoding="utf-8")

    units = list(iter_units(path))

    assert [u.text for u in units] == ["first para\nstill first", "second para"]
    assert [u.index for u in units] == [1, 4]
    assert units[0].kind == KIND_LINE


def test_markdown_is_read_the_same_way_as_text(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# heading\n\nbody text\n", encoding="utf-8")

    assert [u.index for u in iter_units(path)] == [1, 3]


def test_whitespace_only_text_yields_nothing(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n\n  \n", encoding="utf-8")

    assert list(iter_units(path)) == []


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "notes.docx"
    path.write_text("hello", encoding="utf-8")

    with pytest.raises(UnsupportedFileType):
        list(iter_units(path))


def test_unit_kind_is_known_without_reading_the_file():
    assert unit_kind("a.csv") == KIND_ROW
    assert unit_kind("a.json") == KIND_ROW
    assert unit_kind("a.jsonl") == KIND_LINE
    assert unit_kind("a.txt") == KIND_LINE
    assert unit_kind("a.pdf") == KIND_PAGE


def test_is_supported_matches_the_extension_set():
    assert is_supported("a.csv")
    assert is_supported("A.PDF")
    assert not is_supported("a.docx")


def test_csv_columns_returns_the_header(tmp_path):
    path = tmp_path / "docs.csv"
    path.write_text("passage,id\nx,1\n", encoding="utf-8")

    assert csv_columns(path) == ["passage", "id"]


def test_csv_columns_is_none_for_other_formats(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")

    assert csv_columns(path) is None


def test_unit_label_names_the_locator():
    from src.ingestion.units import Unit

    assert Unit(index=42, kind=KIND_ROW, text="x").label == "row 42"
    assert Unit(index=7, kind=KIND_PAGE, text="x").label == "page 7"
