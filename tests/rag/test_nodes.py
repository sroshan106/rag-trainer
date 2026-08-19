from langchain_core.documents import Document

from src.rag import nodes


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

    result = nodes.generate_node({"query": "what?", "graded_docs": docs})

    assert result["answer"] == "the answer"
    assert "fact one" in fake_llm.last_prompt
    assert "what?" in fake_llm.last_prompt


def test_generate_node_strips_a_thinking_block(monkeypatch):
    # Qwen3's Ollama template primes the prompt with <think> but never emits
    # the opening tag in the response -- only the closing one comes back.
    thinking_output = "reasoning about the fact...\n</think>\n\nthe answer"
    monkeypatch.setattr(nodes, "_get_llm", lambda model=None: FakeLLM(thinking_output))
    docs = [Document(page_content="fact one", metadata={"source": "a.txt"})]

    result = nodes.generate_node({"query": "what?", "graded_docs": docs})

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


def test_generate_node_collects_sources(monkeypatch):
    monkeypatch.delenv("RAG_CITATIONS", raising=False)
    monkeypatch.setattr(nodes, "_get_llm", lambda model=None: FakeLLM("the answer"))
    docs = [
        Document(page_content="fact one", metadata={"source": "a.txt"}),
        Document(page_content="fact two", metadata={"source": "a.txt"}),
        Document(page_content="fact three", metadata={"source": "b.txt"}),
    ]

    result = nodes.generate_node({"query": "what?", "graded_docs": docs})

    assert result["sources"] == ["a.txt", "b.txt"]


def test_generate_node_skips_sources_when_disabled(monkeypatch):
    monkeypatch.setenv("RAG_CITATIONS", "false")
    monkeypatch.setattr(nodes, "_get_llm", lambda model=None: FakeLLM("the answer"))
    docs = [Document(page_content="fact one", metadata={"source": "a.txt"})]

    result = nodes.generate_node({"query": "what?", "graded_docs": docs})

    assert result["answer"] == "the answer"
    assert result["sources"] == []


def test_generate_node_uses_the_requested_model(monkeypatch):
    seen = []
    monkeypatch.setattr(
        nodes, "_get_llm", lambda model=None: seen.append(model) or FakeLLM("ok")
    )
    docs = [Document(page_content="fact", metadata={"source": "a.txt"})]

    nodes.generate_node({"query": "q", "model": "qwen3:4b", "graded_docs": docs})

    assert seen == ["qwen3:4b"]


def test_generate_node_defaults_model_when_absent(monkeypatch):
    seen = []
    monkeypatch.setattr(
        nodes, "_get_llm", lambda model=None: seen.append(model) or FakeLLM("ok")
    )
    docs = [Document(page_content="fact", metadata={"source": "a.txt"})]

    nodes.generate_node({"query": "q", "graded_docs": docs})

    assert seen == [nodes.MODEL]


def test_generate_node_fallback_has_empty_sources():
    result = nodes.generate_node({"query": "q", "graded_docs": []})

    assert result["sources"] == []


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
