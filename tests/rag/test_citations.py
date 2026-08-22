from langchain_core.documents import Document

from src.rag.citations import citations_enabled, collect_citations, format_citations


def _doc(text="x", **metadata):
    base = {
        "file_id": "file-1",
        "filename": "corpus.csv",
        "unit_kind": "row",
        "unit_index": 42,
    }
    return Document(page_content=text, metadata={**base, **metadata})


def test_citation_names_the_file_and_the_unit():
    citation = collect_citations([_doc()])[0]

    assert citation["filename"] == "corpus.csv"
    assert citation["file_id"] == "file-1"
    assert citation["unit_index"] == 42
    assert citation["label"] == "row 42"


def test_two_chunks_of_the_same_unit_cite_it_once():
    citations = collect_citations([_doc("first half"), _doc("second half")])

    assert len(citations) == 1


def test_same_index_in_different_files_stays_distinct():
    citations = collect_citations([_doc(), _doc(file_id="file-2", filename="other.csv")])

    assert len(citations) == 2


def test_retrieval_rank_order_is_preserved():
    citations = collect_citations(
        [_doc(unit_index=9), _doc(unit_index=3), _doc(unit_index=7)]
    )

    assert [c["unit_index"] for c in citations] == [9, 3, 7]


def test_chunks_without_provenance_are_dropped():
    citations = collect_citations([_doc(file_id=None), _doc(unit_index=None), _doc()])

    assert len(citations) == 1
    assert citations[0]["unit_index"] == 42


def test_a_url_carried_by_the_source_data_rides_along():
    citation = collect_citations([_doc(url="https://example.com/a")])[0]

    assert citation["url"] == "https://example.com/a"


def test_url_is_none_when_the_source_had_no_link():
    assert collect_citations([_doc()])[0]["url"] is None


def test_pdf_pages_are_labelled_as_pages():
    citation = collect_citations([_doc(unit_kind="page", unit_index=7)])[0]

    assert citation["label"] == "page 7"


def test_format_citations_numbers_them():
    rendered = format_citations(collect_citations([_doc()]))

    assert "[1] corpus.csv -- row 42" in rendered


def test_format_citations_appends_a_url_when_present():
    rendered = format_citations(collect_citations([_doc(url="https://example.com/a")]))

    assert "(https://example.com/a)" in rendered


def test_format_citations_of_nothing_is_empty():
    assert format_citations([]) == ""


def test_citations_are_enabled_by_default(monkeypatch):
    monkeypatch.delenv("RAG_CITATIONS", raising=False)

    assert citations_enabled() is True


def test_citations_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("RAG_CITATIONS", "false")

    assert citations_enabled() is False
