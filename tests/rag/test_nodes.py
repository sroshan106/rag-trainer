import pytest
from langchain_core.documents import Document

from src.rag import generate, grade, models, retrieve
from src.vectorstore import rerank as rerank_module


class FakeVectorstore:
    def __init__(self, scored_docs):
        self.scored_docs = scored_docs
        self.last_query = None
        self.last_k = None

    def similarity_search_with_relevance_scores(self, query, k):
        self.last_query = query
        self.last_k = k
        return self.scored_docs


def _scored(content, score, source="a.txt"):
    return Document(
        page_content=content,
        metadata={"source": source, grade.SCORE_KEY: score},
    )


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, content):
        self.content = content
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return FakeResponse(self.content)


def _dense_only(monkeypatch):
    monkeypatch.setenv("RAG_HYBRID", "false")


def _no_rerank(monkeypatch):
    monkeypatch.setenv("RAG_RERANK", "false")


class FakeCrossEncoder:

    def __init__(self, order):
        self.order = order
        self.pairs = None

    def predict(self, pairs):
        self.pairs = pairs
        return [float(len(self.order) - self.order.index(text)) for _, text in pairs]


def _cited(text, unit_index, file_id="file-1"):
    return Document(
        page_content=text,
        metadata={
            "file_id": file_id,
            "filename": "corpus.csv",
            "unit_kind": "row",
            "unit_index": unit_index,
        },
    )


def _confidence_doc(dense=None, rerank_score=None):
    metadata = {}
    if dense is not None:
        metadata[grade.SCORE_KEY] = dense
    if rerank_score is not None:
        metadata[rerank_module.RERANK_SCORE_KEY] = rerank_score
    return Document(page_content="x", metadata=metadata)


def test_retrieve_node_populates_docs(monkeypatch):
    _dense_only(monkeypatch)
    _no_rerank(monkeypatch)
    doc = Document(page_content="a", metadata={})
    fake_store = FakeVectorstore([(doc, 0.9)])
    monkeypatch.setattr(retrieve, "get_vectorstore", lambda: fake_store)

    result = retrieve.retrieve_node({"query": "q"})

    assert result["retrieved_docs"] == [doc]
    assert fake_store.last_query == "q"
    assert fake_store.last_k == retrieve.RETRIEVE_K


def test_retrieve_node_stamps_relevance_score(monkeypatch):
    _dense_only(monkeypatch)
    _no_rerank(monkeypatch)
    doc = Document(page_content="a", metadata={})
    monkeypatch.setattr(retrieve, "get_vectorstore", lambda: FakeVectorstore([(doc, 0.73)]))

    result = retrieve.retrieve_node({"query": "q"})

    assert result["retrieved_docs"][0].metadata[grade.SCORE_KEY] == 0.73


def test_grade_node_drops_blank_chunks():
    docs = [
        _scored("keep me", 0.9),
        _scored("   ", 0.9),
        _scored("", 0.9),
    ]

    result = grade.grade_node({"retrieved_docs": docs})

    assert len(result["graded_docs"]) == 1
    assert result["graded_docs"][0].page_content == "keep me"


def test_grade_node_drops_chunks_below_absolute_floor():
    docs = [
        _scored("weak one", grade.RELEVANCE_FLOOR - 0.01),
        _scored("weak two", grade.RELEVANCE_FLOOR - 0.05),
    ]

    result = grade.grade_node({"retrieved_docs": docs})

    assert result["graded_docs"] == []


def test_grade_node_keeps_chunks_at_the_floor():
    docs = [_scored("exactly at floor", grade.RELEVANCE_FLOOR)]

    result = grade.grade_node({"retrieved_docs": docs})

    assert len(result["graded_docs"]) == 1


def test_grade_node_drops_chunks_far_weaker_than_best_hit():
    strong = 0.95
    docs = [
        _scored("strong", strong),
        _scored("relatively weak", strong * grade.RELEVANCE_RATIO - 0.01),
    ]

    result = grade.grade_node({"retrieved_docs": docs})

    assert [d.page_content for d in result["graded_docs"]] == ["strong"]


def test_grade_node_keeps_comparable_chunks():
    docs = [_scored("strong", 0.95), _scored("also strong", 0.93)]

    result = grade.grade_node({"retrieved_docs": docs})

    assert len(result["graded_docs"]) == 2


def test_grade_node_empty_retrieval_returns_empty():
    assert grade.grade_node({"retrieved_docs": []})["graded_docs"] == []


def test_generate_node_returns_fallback_when_no_graded_docs():
    result = generate.generate_node({"query": "q", "graded_docs": []})

    assert "don't have enough context" in result["answer"]


def test_generate_node_calls_llm_with_formatted_context(monkeypatch):
    fake_llm = FakeLLM("the answer")
    monkeypatch.setattr(generate, "get_llm", lambda model=None: fake_llm)
    docs = [Document(page_content="fact one", metadata={"source": "a.txt"})]

    result = generate.generate_node({"query": "what?", "model": "llama3.2:3b", "graded_docs": docs})

    assert result["answer"] == "the answer"
    assert "fact one" in fake_llm.last_prompt
    assert "what?" in fake_llm.last_prompt


def test_num_ctx_for_other_models_uses_the_default():
    assert models.num_ctx_for("llama3.2:3b") == models.NUM_CTX


def test_generate_node_collects_citations(monkeypatch):
    monkeypatch.delenv("RAG_CITATIONS", raising=False)
    monkeypatch.setattr(generate, "get_llm", lambda model=None: FakeLLM("the answer"))
    docs = [_cited("fact one", 1), _cited("fact two", 1), _cited("fact three", 2)]

    result = generate.generate_node({"query": "what?", "model": "llama3.2:3b", "graded_docs": docs})

    assert [c["unit_index"] for c in result["citations"]] == [1, 2]
    assert result["refused"] is False


def test_generate_node_skips_citations_when_disabled(monkeypatch):
    monkeypatch.setenv("RAG_CITATIONS", "false")
    monkeypatch.setattr(generate, "get_llm", lambda model=None: FakeLLM("the answer"))
    docs = [_cited("fact one", 1)]

    result = generate.generate_node({"query": "what?", "model": "llama3.2:3b", "graded_docs": docs})

    assert result["answer"] == "the answer"
    assert result["citations"] == []
    assert result["refused"] is False


def test_generate_node_uses_the_requested_model(monkeypatch):
    seen = []
    monkeypatch.setattr(
        generate, "get_llm", lambda model=None: seen.append(model) or FakeLLM("ok")
    )
    docs = [Document(page_content="fact", metadata={"source": "a.txt"})]

    generate.generate_node({"query": "q", "model": "qwen2.5:3b", "graded_docs": docs})

    assert seen == ["qwen2.5:3b"]


def test_generate_node_raises_when_model_is_absent():
    docs = [Document(page_content="fact", metadata={"source": "a.txt"})]

    with pytest.raises(KeyError):
        generate.generate_node({"query": "q", "graded_docs": docs})


def test_generate_node_fallback_is_marked_refused():
    result = generate.generate_node({"query": "q", "graded_docs": []})

    assert result["citations"] == []
    assert result["refused"] is True
    assert result["confidence"] == 0.0


def test_direct_answer_calls_llm_with_the_raw_query_only(monkeypatch):
    fake_llm = FakeLLM("the answer")
    monkeypatch.setattr(generate, "get_llm", lambda model=None: fake_llm)

    result = generate.direct_answer("llama3.2:3b", "what is the capital of France?")

    assert result["answer"] == "the answer"
    assert fake_llm.last_prompt == "what is the capital of France?"


def test_direct_answer_reports_generate_ms(monkeypatch):
    monkeypatch.setattr(generate, "get_llm", lambda model=None: FakeLLM("ok"))

    result = generate.direct_answer("llama3.2:3b", "q")

    assert isinstance(result["generate_ms"], float)
    assert result["generate_ms"] >= 0


def _lexical(content, score, source="a.txt"):
    return Document(
        page_content=content,
        metadata={"source": source, grade.LEXICAL_KEY: score},
    )


def test_retrieve_node_fuses_lexical_hits(monkeypatch):
    _no_rerank(monkeypatch)
    dense = Document(page_content="dense only", metadata={})
    shared = Document(page_content="in both", metadata={})
    monkeypatch.setattr(
        retrieve, "get_vectorstore", lambda: FakeVectorstore([(dense, 0.7), (shared, 0.6)])
    )
    monkeypatch.setattr(
        retrieve.hybrid.lexical,
        "search",
        lambda query, k, connection=None: [(Document(page_content="in both"), 0.9)],
    )

    docs = retrieve.retrieve_node({"query": "q"})["retrieved_docs"]

    assert docs[0].page_content == "in both"
    assert docs[0].metadata[grade.LEXICAL_KEY] == 0.9
    assert docs[0].metadata[grade.SCORE_KEY] == 0.6


def test_grade_node_keeps_lexical_hit_below_dense_floor():
    docs = [_lexical("exact phrase match", 0.02)]
    docs[0].metadata[grade.SCORE_KEY] = grade.RELEVANCE_FLOOR - 0.1

    result = grade.grade_node({"retrieved_docs": docs})

    assert result["graded_docs"] == docs


def test_grade_node_refuses_when_no_lexical_and_dense_below_floor():
    docs = [_scored("weak", grade.RELEVANCE_FLOOR - 0.1)]

    assert grade.grade_node({"retrieved_docs": docs})["graded_docs"] == []


def test_grade_node_dense_cutoff_ignores_lexical_hits():
    strong_lexical = _lexical("phrase", 0.9)
    strong_lexical.metadata[grade.SCORE_KEY] = 0.99
    dense = _scored("dense", 0.7)

    graded = grade.grade_node({"retrieved_docs": [strong_lexical, dense]})["graded_docs"]

    assert dense in graded


def test_retrieve_node_fetches_wide_and_reranks_down(monkeypatch):
    _dense_only(monkeypatch)
    monkeypatch.setenv("RAG_RERANK", "true")
    docs = [Document(page_content=f"d{i}", metadata={}) for i in range(8)]
    fake_store = FakeVectorstore([(d, 0.9 - i * 0.01) for i, d in enumerate(docs)])
    monkeypatch.setattr(retrieve, "get_vectorstore", lambda: fake_store)
    encoder = FakeCrossEncoder(["d7"] + [f"d{i}" for i in range(7)])
    monkeypatch.setattr(retrieve.rerank, "_get_model", lambda: encoder)

    result = retrieve.retrieve_node({"query": "q"})

    assert fake_store.last_k == retrieve.FETCH_K
    assert len(result["retrieved_docs"]) == retrieve.RETRIEVE_K
    assert result["retrieved_docs"][0].page_content == "d7"


def test_rerank_preserves_component_scores(monkeypatch):
    _dense_only(monkeypatch)
    monkeypatch.setenv("RAG_RERANK", "true")
    doc = Document(page_content="a", metadata={})
    monkeypatch.setattr(retrieve, "get_vectorstore", lambda: FakeVectorstore([(doc, 0.73)]))
    monkeypatch.setattr(retrieve.rerank, "_get_model", lambda: FakeCrossEncoder(["a"]))

    kept = retrieve.retrieve_node({"query": "q"})["retrieved_docs"][0]

    assert kept.metadata[grade.SCORE_KEY] == 0.73
    assert 0.0 < kept.metadata[retrieve.rerank.RERANK_SCORE_KEY] < 1.0


class FakeStreamingLLM:

    def __init__(self, chunks):
        self.chunks = chunks
        self.last_prompt = None
        self.closed = False

    def stream(self, prompt):
        self.last_prompt = prompt
        try:
            for chunk in self.chunks:
                yield FakeResponse(chunk)
        except GeneratorExit:
            self.closed = True
            raise


def _drain(generator):
    yielded = []
    while True:
        try:
            yielded.append(next(generator))
        except StopIteration as stop:
            return yielded, stop.value


def test_generate_stream_yields_tokens_and_returns_the_assembled_answer(monkeypatch):
    monkeypatch.delenv("RAG_CITATIONS", raising=False)
    fake_llm = FakeStreamingLLM(["the ", "answer"])
    monkeypatch.setattr(generate, "get_llm", lambda model=None: fake_llm)
    docs = [_cited("fact one", 1)]

    tokens, result = _drain(generate.generate_stream({"query": "what?", "model": "llama3.2:3b", "graded_docs": docs}))

    assert tokens == ["the ", "answer"]
    assert result["answer"] == "the answer"
    assert [c["unit_index"] for c in result["citations"]] == [1]
    assert result["refused"] is False
    assert "fact one" in fake_llm.last_prompt
    assert "what?" in fake_llm.last_prompt


def test_generate_stream_refuses_without_graded_docs():
    tokens, result = _drain(generate.generate_stream({"query": "q", "graded_docs": []}))

    assert tokens == [generate.REFUSAL_ANSWER]
    assert result == {
        "answer": generate.REFUSAL_ANSWER,
        "citations": [],
        "refused": True,
        "confidence": 0.0,
    }


def test_generate_stream_uses_the_requested_model(monkeypatch):
    seen = []
    monkeypatch.setattr(
        generate,
        "get_llm",
        lambda model=None: seen.append(model) or FakeStreamingLLM(["ok"]),
    )
    docs = [Document(page_content="fact", metadata={"source": "a.txt"})]

    _drain(generate.generate_stream({"query": "q", "model": "qwen2.5:3b", "graded_docs": docs}))

    assert seen == ["qwen2.5:3b"]


def test_generate_stream_closing_early_stops_the_llm(monkeypatch):
    fake_llm = FakeStreamingLLM(["one ", "two ", "three"])
    monkeypatch.setattr(generate, "get_llm", lambda model=None: fake_llm)
    docs = [Document(page_content="fact", metadata={"source": "a.txt"})]

    generation = generate.generate_stream({"query": "q", "model": "llama3.2:3b", "graded_docs": docs})
    assert next(generation) == "one "
    generation.close()

    assert fake_llm.closed is True


def test_generate_stream_skips_citations_when_disabled(monkeypatch):
    monkeypatch.setenv("RAG_CITATIONS", "false")
    monkeypatch.setattr(generate, "get_llm", lambda model=None: FakeStreamingLLM(["a"]))
    docs = [_cited("fact", 1)]

    _, result = _drain(generate.generate_stream({"query": "q", "model": "llama3.2:3b", "graded_docs": docs}))

    assert result["citations"] == []
    assert result["refused"] is False


def test_prompt_for_requires_a_model():
    docs = [Document(page_content="fact one", metadata={"source": "a.txt"})]

    with pytest.raises(KeyError):
        generate.prompt_for({"query": "what?", "graded_docs": docs})


def test_prompt_for_builds_the_prompt_for_the_given_model():
    docs = [Document(page_content="fact one", metadata={"source": "a.txt"})]

    model, prompt = generate.prompt_for(
        {"query": "what?", "model": "llama3.2:3b", "graded_docs": docs}
    )

    assert model == "llama3.2:3b"
    assert "fact one" in prompt
    assert "what?" in prompt


def test_refusal_is_the_shared_constant():
    assert generate._refusal() == {
        "answer": generate.REFUSAL_ANSWER,
        "citations": [],
        "refused": True,
        "confidence": 0.0,
    }


def test_confidence_prefers_the_cross_encoder_score():
    docs = [
        _confidence_doc(dense=0.60, rerank_score=0.91),
        _confidence_doc(dense=0.99, rerank_score=0.40),
    ]

    assert generate.confidence_of(docs) == 0.91


def test_confidence_falls_back_to_dense_when_reranking_is_off():
    assert generate.confidence_of([_confidence_doc(dense=0.73)]) == 0.73


def test_confidence_takes_the_best_chunk_not_the_average():
    docs = [
        _confidence_doc(dense=0.9),
        _confidence_doc(dense=0.1),
        _confidence_doc(dense=0.1),
    ]

    assert generate.confidence_of(docs) == 0.9


def test_confidence_of_nothing_is_zero():
    assert generate.confidence_of([]) == 0.0


def test_confidence_ignores_a_lexical_only_hit_with_no_comparable_score():
    lexical_only = Document(page_content="x", metadata={grade.LEXICAL_KEY: 4.2})

    assert generate.confidence_of([lexical_only]) == 0.0
