from langchain_core.documents import Document

from src.vectorstore import hybrid


class FakeVectorstore:
    def __init__(self, scored):
        self.scored = scored

    def similarity_search_with_relevance_scores(self, query, k):
        return self.scored


def _patch_lexical(monkeypatch, results):
    monkeypatch.setattr(
        hybrid.lexical, "search", lambda query, k, connection=None: results
    )


def test_documents_in_both_lists_outrank_single_list_hits(monkeypatch):
    dense_top = Document(page_content="dense winner", metadata={})
    both = Document(page_content="both", metadata={})
    store = FakeVectorstore([(dense_top, 0.9), (both, 0.5)])
    _patch_lexical(monkeypatch, [(Document(page_content="both"), 0.1)])

    ranked = hybrid.retrieve(store, "q", k=5)

    assert [d.page_content for d in ranked][0] == "both"


def test_lexical_only_hit_is_retained(monkeypatch):
    dense = Document(page_content="dense", metadata={})
    store = FakeVectorstore([(dense, 0.9)])
    _patch_lexical(monkeypatch, [(Document(page_content="lexical only"), 0.4)])

    contents = {d.page_content for d in hybrid.retrieve(store, "q", k=5)}

    assert contents == {"dense", "lexical only"}


def test_component_scores_are_stamped(monkeypatch):
    doc = Document(page_content="both", metadata={})
    store = FakeVectorstore([(doc, 0.62)])
    _patch_lexical(monkeypatch, [(Document(page_content="both"), 0.31)])

    ranked = hybrid.retrieve(store, "q", k=5)

    assert ranked[0].metadata[hybrid.DENSE_SCORE_KEY] == 0.62
    assert ranked[0].metadata[hybrid.LEXICAL_SCORE_KEY] == 0.31
    assert ranked[0].metadata[hybrid.FUSION_SCORE_KEY] > 0


def test_empty_lexical_list_falls_back_to_dense_order(monkeypatch):
    """Off-topic queries match no tsvector, so fusion must degrade to dense."""
    first = Document(page_content="first", metadata={})
    second = Document(page_content="second", metadata={})
    store = FakeVectorstore([(first, 0.45), (second, 0.44)])
    _patch_lexical(monkeypatch, [])

    ranked = hybrid.retrieve(store, "q", k=5)

    assert [d.page_content for d in ranked] == ["first", "second"]


def test_result_is_truncated_to_k(monkeypatch):
    store = FakeVectorstore(
        [(Document(page_content=f"d{i}", metadata={}), 0.9 - i / 100) for i in range(5)]
    )
    _patch_lexical(
        monkeypatch, [(Document(page_content=f"l{i}"), 0.5) for i in range(5)]
    )

    assert len(hybrid.retrieve(store, "q", k=5)) == 5


def test_rrf_ties_break_toward_the_lexical_hit(monkeypatch):
    """Swapped ranks score identically; exact-phrase evidence wins the tie."""
    a = Document(page_content="a", metadata={})
    b = Document(page_content="b", metadata={})
    store = FakeVectorstore([(a, 0.7), (b, 0.6)])
    _patch_lexical(
        monkeypatch,
        [(Document(page_content="b"), 0.9), (Document(page_content="a"), 0.1)],
    )

    ranked = hybrid.retrieve(store, "q", k=5)

    assert ranked[0].page_content == "b"
