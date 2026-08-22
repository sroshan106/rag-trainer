import math

from langchain_core.documents import Document

from src.vectorstore import rerank


class FakeCrossEncoder:

    def __init__(self, logits):
        self.logits = logits
        self.pairs = None

    def predict(self, pairs):
        self.pairs = pairs
        return self.logits[: len(pairs)]


def _docs(*texts):
    return [Document(page_content=t, metadata={}) for t in texts]


def _patch(monkeypatch, encoder):
    monkeypatch.setattr(rerank, "_get_model", lambda: encoder)
    return encoder


def test_reorders_by_cross_encoder_score(monkeypatch):
    _patch(monkeypatch, FakeCrossEncoder([-2.0, 5.0, 1.0]))

    ranked = rerank.rerank("q", _docs("low", "high", "mid"), k=3)

    assert [d.page_content for d in ranked] == ["high", "mid", "low"]


def test_truncates_to_k(monkeypatch):
    _patch(monkeypatch, FakeCrossEncoder([1.0, 2.0, 3.0, 4.0]))

    ranked = rerank.rerank("q", _docs("a", "b", "c", "d"), k=2)

    assert [d.page_content for d in ranked] == ["d", "c"]


def test_every_candidate_is_scored_before_truncation(monkeypatch):
    docs = _docs("a", "b", "c")
    _patch(monkeypatch, FakeCrossEncoder([3.0, 1.0, 2.0]))

    rerank.rerank("q", docs, k=1)

    assert all(rerank.RERANK_SCORE_KEY in d.metadata for d in docs)


def test_scores_are_squashed_to_unit_interval(monkeypatch):
    _patch(monkeypatch, FakeCrossEncoder([0.0, 10.0, -10.0]))

    ranked = rerank.rerank("q", _docs("zero", "big", "small"), k=3)
    scores = {d.page_content: d.metadata[rerank.RERANK_SCORE_KEY] for d in ranked}

    assert math.isclose(scores["zero"], 0.5)
    assert 0.0 < scores["small"] < scores["zero"] < scores["big"] < 1.0


def test_query_is_paired_with_each_document(monkeypatch):
    encoder = _patch(monkeypatch, FakeCrossEncoder([1.0, 1.0]))

    rerank.rerank("what is a bullet kin", _docs("a", "b"), k=2)

    assert encoder.pairs == [("what is a bullet kin", "a"), ("what is a bullet kin", "b")]


def test_empty_candidate_list_does_not_load_the_model(monkeypatch):
    def explode():
        raise AssertionError("model loaded for an empty list")

    monkeypatch.setattr(rerank, "_get_model", explode)

    assert rerank.rerank("q", [], k=5) == []


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("RAG_RERANK", "false")
    assert rerank.rerank_enabled() is False


def test_enabled_by_default(monkeypatch):
    monkeypatch.delenv("RAG_RERANK", raising=False)
    assert rerank.rerank_enabled() is True
