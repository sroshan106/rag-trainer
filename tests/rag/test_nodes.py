import pytest
from langchain_core.documents import Document

from src.rag import nodes
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
    """A doc as grade_node sees it: score already stamped by retrieve_node."""
    return Document(
        page_content=content,
        metadata={"source": source, nodes.SCORE_KEY: score},
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
    """Pin retrieval to the dense path, which needs no live full-text index."""
    monkeypatch.setenv("RAG_HYBRID", "false")


def _no_rerank(monkeypatch):
    """Assert on raw retrieval order, before the cross-encoder reorders it."""
    monkeypatch.setenv("RAG_RERANK", "false")


class FakeCrossEncoder:
    """Scores by position in a caller-supplied ranking of document texts."""

    def __init__(self, order):
        self.order = order
        self.pairs = None

    def predict(self, pairs):
        self.pairs = pairs
        return [float(len(self.order) - self.order.index(text)) for _, text in pairs]


def _cited(text, unit_index, file_id="file-1"):
    """A document carrying the provenance ingest stamps on every chunk."""
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
    """A doc carrying only the score keys confidence_of reads."""
    metadata = {}
    if dense is not None:
        metadata[nodes.SCORE_KEY] = dense
    if rerank_score is not None:
        metadata[rerank_module.RERANK_SCORE_KEY] = rerank_score
    return Document(page_content="x", metadata=metadata)


def test_retrieve_node_populates_docs(monkeypatch):
    _dense_only(monkeypatch)
    _no_rerank(monkeypatch)
    doc = Document(page_content="a", metadata={})
    fake_store = FakeVectorstore([(doc, 0.9)])
    monkeypatch.setattr(nodes, "_get_vectorstore", lambda: fake_store)

    result = nodes.retrieve_node({"query": "q"})

    assert result["retrieved_docs"] == [doc]
    assert fake_store.last_query == "q"
    assert fake_store.last_k == nodes.RETRIEVE_K


def test_retrieve_node_stamps_relevance_score(monkeypatch):
    _dense_only(monkeypatch)
    _no_rerank(monkeypatch)
    doc = Document(page_content="a", metadata={})
    monkeypatch.setattr(nodes, "_get_vectorstore", lambda: FakeVectorstore([(doc, 0.73)]))

    result = nodes.retrieve_node({"query": "q"})

    assert result["retrieved_docs"][0].metadata[nodes.SCORE_KEY] == 0.73


def test_grade_node_drops_blank_chunks():
    docs = [
        _scored("keep me", 0.9),
        _scored("   ", 0.9),
        _scored("", 0.9),
    ]

    result = nodes.grade_node({"retrieved_docs": docs})

    assert len(result["graded_docs"]) == 1
    assert result["graded_docs"][0].page_content == "keep me"


def test_grade_node_drops_chunks_below_absolute_floor():
    # Every hit is weak: an off-topic query must end up refusing, not citing
    # the five least-bad chunks in the collection.
    docs = [
        _scored("weak one", nodes.RELEVANCE_FLOOR - 0.01),
        _scored("weak two", nodes.RELEVANCE_FLOOR - 0.05),
    ]

    result = nodes.grade_node({"retrieved_docs": docs})

    assert result["graded_docs"] == []


def test_grade_node_keeps_chunks_at_the_floor():
    docs = [_scored("exactly at floor", nodes.RELEVANCE_FLOOR)]

    result = nodes.grade_node({"retrieved_docs": docs})

    assert len(result["graded_docs"]) == 1


def test_grade_node_drops_chunks_far_weaker_than_best_hit():
    # Both clear the floor, but the second is noise next to the top hit.
    strong = 0.95
    docs = [
        _scored("strong", strong),
        _scored("relatively weak", strong * nodes.RELEVANCE_RATIO - 0.01),
    ]

    result = nodes.grade_node({"retrieved_docs": docs})

    assert [d.page_content for d in result["graded_docs"]] == ["strong"]


def test_grade_node_keeps_comparable_chunks():
    docs = [_scored("strong", 0.95), _scored("also strong", 0.93)]

    result = nodes.grade_node({"retrieved_docs": docs})

    assert len(result["graded_docs"]) == 2


def test_grade_node_empty_retrieval_returns_empty():
    assert nodes.grade_node({"retrieved_docs": []})["graded_docs"] == []


def test_generate_node_returns_fallback_when_no_graded_docs():
    result = nodes.generate_node({"query": "q", "graded_docs": []})

    assert "don't have enough context" in result["answer"]


def test_generate_node_calls_llm_with_formatted_context(monkeypatch):
    fake_llm = FakeLLM("the answer")
    monkeypatch.setattr(nodes, "_get_llm", lambda model=None: fake_llm)
    docs = [Document(page_content="fact one", metadata={"source": "a.txt"})]

    result = nodes.generate_node({"query": "what?", "model": "llama3.2:3b", "graded_docs": docs})

    assert result["answer"] == "the answer"
    assert "fact one" in fake_llm.last_prompt
    assert "what?" in fake_llm.last_prompt


def test_generate_node_strips_a_thinking_block(monkeypatch):
    # Qwen3's Ollama template primes the prompt with <think> but never emits
    # the opening tag in the response -- only the closing one comes back.
    thinking_output = "reasoning about the fact...\n</think>\n\nthe answer"
    monkeypatch.setattr(nodes, "_get_llm", lambda model=None: FakeLLM(thinking_output))
    docs = [Document(page_content="fact one", metadata={"source": "a.txt"})]

    result = nodes.generate_node({"query": "what?", "model": "llama3.2:3b", "graded_docs": docs})

    assert result["answer"] == "the answer"


def test_strip_thinking_passes_through_plain_text():
    assert nodes._strip_thinking("just an answer") == "just an answer"


def test_num_ctx_for_qwen3_uses_the_smaller_gpu_fitting_context():
    assert nodes._num_ctx_for("qwen3:4b") == nodes.QWEN3_NUM_CTX


def test_num_ctx_for_other_models_uses_the_default():
    assert nodes._num_ctx_for("llama3.2:3b") == nodes.NUM_CTX


def test_generate_node_appends_no_think_for_a_thinking_model(monkeypatch):
    fake_llm = FakeLLM("the answer")
    monkeypatch.setattr(nodes, "_get_llm", lambda model=None: fake_llm)
    docs = [Document(page_content="fact one", metadata={"source": "a.txt"})]

    nodes.generate_node({"query": "what?", "model": "qwen3:4b", "graded_docs": docs})

    assert fake_llm.last_prompt.endswith("/no_think")


def test_generate_node_does_not_append_no_think_for_llama(monkeypatch):
    fake_llm = FakeLLM("the answer")
    monkeypatch.setattr(nodes, "_get_llm", lambda model=None: fake_llm)
    docs = [Document(page_content="fact one", metadata={"source": "a.txt"})]

    nodes.generate_node({"query": "what?", "model": "llama3.2:3b", "graded_docs": docs})

    assert "/no_think" not in fake_llm.last_prompt


def test_generate_node_collects_citations(monkeypatch):
    monkeypatch.delenv("RAG_CITATIONS", raising=False)
    monkeypatch.setattr(nodes, "_get_llm", lambda model=None: FakeLLM("the answer"))
    docs = [_cited("fact one", 1), _cited("fact two", 1), _cited("fact three", 2)]

    result = nodes.generate_node({"query": "what?", "model": "llama3.2:3b", "graded_docs": docs})

    assert [c["unit_index"] for c in result["citations"]] == [1, 2]
    assert result["refused"] is False


def test_generate_node_skips_citations_when_disabled(monkeypatch):
    monkeypatch.setenv("RAG_CITATIONS", "false")
    monkeypatch.setattr(nodes, "_get_llm", lambda model=None: FakeLLM("the answer"))
    docs = [_cited("fact one", 1)]

    result = nodes.generate_node({"query": "what?", "model": "llama3.2:3b", "graded_docs": docs})

    assert result["answer"] == "the answer"
    assert result["citations"] == []
    # Turning citations off must not turn the answer into a refusal.
    assert result["refused"] is False


def test_generate_node_uses_the_requested_model(monkeypatch):
    seen = []
    monkeypatch.setattr(
        nodes, "_get_llm", lambda model=None: seen.append(model) or FakeLLM("ok")
    )
    docs = [Document(page_content="fact", metadata={"source": "a.txt"})]

    nodes.generate_node({"query": "q", "model": "qwen3:4b", "graded_docs": docs})

    assert seen == ["qwen3:4b"]


def test_generate_node_raises_when_model_is_absent():
    # No fallback: graph.py is responsible for resolving a real model before
    # a node ever runs, so a missing "model" key is a caller bug, not
    # something this node papers over.
    docs = [Document(page_content="fact", metadata={"source": "a.txt"})]

    with pytest.raises(KeyError):
        nodes.generate_node({"query": "q", "graded_docs": docs})


def test_generate_node_fallback_is_marked_refused():
    result = nodes.generate_node({"query": "q", "graded_docs": []})

    assert result["citations"] == []
    assert result["refused"] is True
    assert result["confidence"] == 0.0


def test_direct_answer_calls_llm_with_the_raw_query_only(monkeypatch):
    fake_llm = FakeLLM("the answer")
    monkeypatch.setattr(nodes, "_get_llm", lambda model=None: fake_llm)

    result = nodes.direct_answer("llama3.2:3b", "what is the capital of France?")

    assert result["answer"] == "the answer"
    assert fake_llm.last_prompt == "what is the capital of France?"


def test_direct_answer_appends_no_think_for_a_thinking_model(monkeypatch):
    fake_llm = FakeLLM("the answer")
    monkeypatch.setattr(nodes, "_get_llm", lambda model=None: fake_llm)

    nodes.direct_answer("qwen3:4b", "q")

    assert fake_llm.last_prompt.endswith("/no_think")


def test_direct_answer_strips_a_thinking_block(monkeypatch):
    thinking_output = "reasoning...\n</think>\n\nthe answer"
    monkeypatch.setattr(nodes, "_get_llm", lambda model=None: FakeLLM(thinking_output))

    result = nodes.direct_answer("qwen3:4b", "q")

    assert result["answer"] == "the answer"


def test_direct_answer_reports_generate_ms(monkeypatch):
    monkeypatch.setattr(nodes, "_get_llm", lambda model=None: FakeLLM("ok"))

    result = nodes.direct_answer("llama3.2:3b", "q")

    assert isinstance(result["generate_ms"], float)
    assert result["generate_ms"] >= 0


def _lexical(content, score, source="a.txt"):
    """A doc as the full-text half of hybrid retrieval returns it."""
    return Document(
        page_content=content,
        metadata={"source": source, nodes.LEXICAL_KEY: score},
    )


def test_retrieve_node_fuses_lexical_hits(monkeypatch):
    _no_rerank(monkeypatch)
    dense = Document(page_content="dense only", metadata={})
    shared = Document(page_content="in both", metadata={})
    monkeypatch.setattr(
        nodes, "_get_vectorstore", lambda: FakeVectorstore([(dense, 0.7), (shared, 0.6)])
    )
    monkeypatch.setattr(
        nodes.hybrid.lexical,
        "search",
        lambda query, k, connection=None: [(Document(page_content="in both"), 0.9)],
    )

    docs = nodes.retrieve_node({"query": "q"})["retrieved_docs"]

    # Present in both lists, so RRF ranks it above the higher-scoring dense hit.
    assert docs[0].page_content == "in both"
    assert docs[0].metadata[nodes.LEXICAL_KEY] == 0.9
    assert docs[0].metadata[nodes.SCORE_KEY] == 0.6


def test_grade_node_keeps_lexical_hit_below_dense_floor():
    """The case dense-only grading got wrong: a real match scoring under the floor."""
    docs = [_lexical("exact phrase match", 0.02)]
    docs[0].metadata[nodes.SCORE_KEY] = nodes.RELEVANCE_FLOOR - 0.1

    result = nodes.grade_node({"retrieved_docs": docs})

    assert result["graded_docs"] == docs


def test_grade_node_refuses_when_no_lexical_and_dense_below_floor():
    docs = [_scored("weak", nodes.RELEVANCE_FLOOR - 0.1)]

    assert nodes.grade_node({"retrieved_docs": docs})["graded_docs"] == []


def test_grade_node_dense_cutoff_ignores_lexical_hits():
    """A strong lexical hit must not raise the bar for dense-only documents."""
    strong_lexical = _lexical("phrase", 0.9)
    strong_lexical.metadata[nodes.SCORE_KEY] = 0.99
    dense = _scored("dense", 0.7)

    graded = nodes.grade_node({"retrieved_docs": [strong_lexical, dense]})["graded_docs"]

    assert dense in graded


def test_retrieve_node_fetches_wide_and_reranks_down(monkeypatch):
    """The whole point of the reranker: recover a chunk buried below RETRIEVE_K."""
    _dense_only(monkeypatch)
    monkeypatch.setenv("RAG_RERANK", "true")
    docs = [Document(page_content=f"d{i}", metadata={}) for i in range(8)]
    fake_store = FakeVectorstore([(d, 0.9 - i * 0.01) for i, d in enumerate(docs)])
    monkeypatch.setattr(nodes, "_get_vectorstore", lambda: fake_store)
    # The cross-encoder disagrees with retrieval: it puts the last-ranked
    # candidate first, which is unreachable without the wide fetch.
    encoder = FakeCrossEncoder(["d7"] + [f"d{i}" for i in range(7)])
    monkeypatch.setattr(nodes.rerank, "_get_model", lambda: encoder)

    result = nodes.retrieve_node({"query": "q"})

    assert fake_store.last_k == nodes.FETCH_K
    assert len(result["retrieved_docs"]) == nodes.RETRIEVE_K
    assert result["retrieved_docs"][0].page_content == "d7"


def test_rerank_preserves_component_scores(monkeypatch):
    """Reranking reorders; it must not strip what grade_node still thresholds on."""
    _dense_only(monkeypatch)
    monkeypatch.setenv("RAG_RERANK", "true")
    doc = Document(page_content="a", metadata={})
    monkeypatch.setattr(nodes, "_get_vectorstore", lambda: FakeVectorstore([(doc, 0.73)]))
    monkeypatch.setattr(nodes.rerank, "_get_model", lambda: FakeCrossEncoder(["a"]))

    kept = nodes.retrieve_node({"query": "q"})["retrieved_docs"][0]

    assert kept.metadata[nodes.SCORE_KEY] == 0.73
    assert 0.0 < kept.metadata[nodes.rerank.RERANK_SCORE_KEY] < 1.0


class FakeStreamingLLM:
    """A ChatOllama stand-in for the streaming path: chunks with ``.content``."""

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
    """Run a generator to exhaustion, returning ``(yielded, return_value)``."""
    yielded = []
    while True:
        try:
            yielded.append(next(generator))
        except StopIteration as stop:
            return yielded, stop.value


def test_think_filter_passes_plain_text_straight_through():
    think = nodes._ThinkFilter()

    assert think.feed("Hello ") == "Hello "
    assert think.feed("world") == "world"


def test_think_filter_suppresses_a_think_block_across_chunk_boundaries():
    think = nodes._ThinkFilter()

    # The opening tag arrives split, so no single chunk ever looks like a tag.
    assert think.feed("<thi") == ""
    assert think.feed("nk>") == ""
    assert think.feed("reasoning about it") == ""
    assert think.feed("</think>the ") == "the "
    # Everything after the close tag streams without further inspection.
    assert think.feed("<think>") == "<think>"


def test_think_filter_releases_text_that_only_looked_like_a_tag():
    think = nodes._ThinkFilter()

    assert think.feed("<") == ""
    assert think.feed("p>hi") == "<p>hi"


def test_generate_stream_yields_tokens_and_returns_the_assembled_answer(monkeypatch):
    monkeypatch.delenv("RAG_CITATIONS", raising=False)
    fake_llm = FakeStreamingLLM(["the ", "answer"])
    monkeypatch.setattr(nodes, "_get_llm", lambda model=None: fake_llm)
    docs = [_cited("fact one", 1)]

    tokens, result = _drain(nodes.generate_stream({"query": "what?", "model": "llama3.2:3b", "graded_docs": docs}))

    assert tokens == ["the ", "answer"]
    assert result["answer"] == "the answer"
    assert [c["unit_index"] for c in result["citations"]] == [1]
    assert result["refused"] is False
    assert "fact one" in fake_llm.last_prompt
    assert "what?" in fake_llm.last_prompt


def test_generate_stream_strips_a_streamed_thinking_block(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "_get_llm",
        lambda model=None: FakeStreamingLLM(["<think>", "hmm", "</think>", "the answer"]),
    )
    docs = [Document(page_content="fact one", metadata={"source": "a.txt"})]

    tokens, result = _drain(nodes.generate_stream({"query": "what?", "model": "llama3.2:3b", "graded_docs": docs}))

    assert tokens == ["the answer"]
    assert result["answer"] == "the answer"


def test_generate_stream_refuses_without_graded_docs():
    tokens, result = _drain(nodes.generate_stream({"query": "q", "graded_docs": []}))

    assert tokens == [nodes.REFUSAL_ANSWER]
    assert result == {
        "answer": nodes.REFUSAL_ANSWER,
        "citations": [],
        "refused": True,
        "confidence": 0.0,
    }


def test_generate_stream_uses_the_requested_model(monkeypatch):
    seen = []
    monkeypatch.setattr(
        nodes,
        "_get_llm",
        lambda model=None: seen.append(model) or FakeStreamingLLM(["ok"]),
    )
    docs = [Document(page_content="fact", metadata={"source": "a.txt"})]

    _drain(nodes.generate_stream({"query": "q", "model": "qwen3:4b", "graded_docs": docs}))

    assert seen == ["qwen3:4b"]


def test_generate_stream_closing_early_stops_the_llm(monkeypatch):
    """Cancellation has to reach Ollama, not just stop the caller reading."""
    fake_llm = FakeStreamingLLM(["one ", "two ", "three"])
    monkeypatch.setattr(nodes, "_get_llm", lambda model=None: fake_llm)
    docs = [Document(page_content="fact", metadata={"source": "a.txt"})]

    generation = nodes.generate_stream({"query": "q", "model": "llama3.2:3b", "graded_docs": docs})
    assert next(generation) == "one "
    generation.close()

    assert fake_llm.closed is True


def test_generate_stream_skips_citations_when_disabled(monkeypatch):
    monkeypatch.setenv("RAG_CITATIONS", "false")
    monkeypatch.setattr(nodes, "_get_llm", lambda model=None: FakeStreamingLLM(["a"]))
    docs = [_cited("fact", 1)]

    _, result = _drain(nodes.generate_stream({"query": "q", "model": "llama3.2:3b", "graded_docs": docs}))

    assert result["citations"] == []
    assert result["refused"] is False


def test_prompt_for_requires_a_model():
    docs = [Document(page_content="fact one", metadata={"source": "a.txt"})]

    with pytest.raises(KeyError):
        nodes._prompt_for({"query": "what?", "graded_docs": docs})


def test_prompt_for_builds_the_prompt_for_the_given_model():
    docs = [Document(page_content="fact one", metadata={"source": "a.txt"})]

    model, prompt = nodes._prompt_for(
        {"query": "what?", "model": "llama3.2:3b", "graded_docs": docs}
    )

    assert model == "llama3.2:3b"
    assert "fact one" in prompt
    assert "what?" in prompt


def test_prompt_for_appends_no_think_for_a_thinking_model():
    docs = [Document(page_content="fact", metadata={"source": "a.txt"})]

    model, prompt = nodes._prompt_for(
        {"query": "q", "model": "qwen3:4b", "graded_docs": docs}
    )

    assert model == "qwen3:4b"
    assert prompt.endswith("/no_think")


def test_refusal_is_the_shared_constant():
    assert nodes._refusal() == {
        "answer": nodes.REFUSAL_ANSWER,
        "citations": [],
        "refused": True,
        "confidence": 0.0,
    }


def test_confidence_prefers_the_cross_encoder_score():
    """It reads query and chunk together; the cosine scores them apart."""
    docs = [
        _confidence_doc(dense=0.60, rerank_score=0.91),
        _confidence_doc(dense=0.99, rerank_score=0.40),
    ]

    assert nodes.confidence_of(docs) == 0.91


def test_confidence_falls_back_to_dense_when_reranking_is_off():
    assert nodes.confidence_of([_confidence_doc(dense=0.73)]) == 0.73


def test_confidence_takes_the_best_chunk_not_the_average():
    """One strong chunk is enough to answer from."""
    docs = [
        _confidence_doc(dense=0.9),
        _confidence_doc(dense=0.1),
        _confidence_doc(dense=0.1),
    ]

    assert nodes.confidence_of(docs) == 0.9


def test_confidence_of_nothing_is_zero():
    assert nodes.confidence_of([]) == 0.0


def test_confidence_ignores_a_lexical_only_hit_with_no_comparable_score():
    lexical_only = Document(page_content="x", metadata={nodes.LEXICAL_KEY: 4.2})

    assert nodes.confidence_of([lexical_only]) == 0.0
