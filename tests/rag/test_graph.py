import pytest
from langchain_core.documents import Document

from src.rag import graph as graph_module
from src.rag.graph import build_graph


def test_build_graph_compiles():
    graph = build_graph()

    assert graph is not None


def test_graph_has_no_retry_edge_back_into_retrieve():
    edges = build_graph().get_graph().edges

    assert not [e for e in edges if e.source == "grade" and e.target == "retrieve"]


class FakeHistory:

    def __init__(self):
        self.completed = None
        self.cancelled = None
        self.failed = None

    def start(self, query, model=None):
        self.query = query
        return "entry-1"

    def complete(self, entry_id, **kwargs):
        self.completed = (entry_id, kwargs)

    def cancel(self, entry_id, partial_answer=""):
        self.cancelled = (entry_id, partial_answer)

    def fail(self, entry_id):
        self.failed = entry_id


def _doc(text, source="a.txt"):
    return Document(page_content=text, metadata={"source": source})


@pytest.fixture
def streaming(monkeypatch):
    fake_history = FakeHistory()
    monkeypatch.setattr(graph_module, "history", fake_history)
    monkeypatch.setattr(
        graph_module,
        "retrieve_node",
        lambda state: {"retrieved_docs": [_doc("one"), _doc("two"), _doc("three")]},
    )
    monkeypatch.setattr(
        graph_module,
        "grade_node",
        lambda state: {"graded_docs": state["retrieved_docs"][:2]},
    )

    def fake_generate_stream(state):
        yield "the "
        yield "answer"
        return {
            "answer": "the answer",
            "citations": [CITATION],
            "refused": False,
            "confidence": 0.87,
        }

    monkeypatch.setattr(graph_module, "generate_stream", fake_generate_stream)
    return fake_history


TEST_MODEL = "llama3.2:3b"

CITATION = {
    "file_id": "file-1",
    "filename": "corpus.csv",
    "unit_kind": "row",
    "unit_index": 42,
    "label": "row 42",
    "url": None,
}


def test_ask_stream_emits_stages_then_tokens_then_done(streaming):
    events = list(graph_module.ask_stream("what?", model=TEST_MODEL))

    assert [e["type"] for e in events] == [
        "stage",
        "stage",
        "stage",
        "token",
        "token",
        "done",
    ]
    stages = [e for e in events if e["type"] == "stage"]
    assert [s["stage"] for s in stages] == ["retrieve", "grade", "generate"]
    assert stages[1]["detail"] == {"retrieved": 3}
    assert stages[2]["detail"] == {"retrieved": 3, "kept": 2}
    assert [e["text"] for e in events if e["type"] == "token"] == ["the ", "answer"]


def test_ask_stream_done_carries_the_answer_and_timings(streaming):
    done = list(graph_module.ask_stream("what?", model=TEST_MODEL))[-1]

    assert done["type"] == "done"
    assert done["id"] == "entry-1"
    assert done["answer"] == "the answer"
    assert done["citations"] == [CITATION]
    assert done["refused"] is False
    assert done["confidence"] == 0.87
    assert done["model"] == TEST_MODEL
    assert isinstance(done["latency_ms"], float)
    assert "rerank_ms" in done and "generate_ms" in done


def test_ask_stream_records_the_completed_answer(streaming):
    list(graph_module.ask_stream("what?", model=TEST_MODEL))

    entry_id, values = streaming.completed
    assert entry_id == "entry-1"
    assert values["answer"] == "the answer"
    assert values["citations"] == [CITATION]
    assert values["confidence"] == 0.87


def test_ask_stream_raises_for_an_unrecognized_model(streaming):
    with pytest.raises(ValueError, match="model is required"):
        list(graph_module.ask_stream("q", model="gpt-4"))


def test_ask_stream_raises_when_no_model_given(streaming):
    with pytest.raises(ValueError, match="model is required"):
        list(graph_module.ask_stream("q"))


def test_ask_stream_passes_a_known_model_through(streaming):
    done = list(graph_module.ask_stream("q", model="qwen2.5:3b"))[-1]

    assert done["model"] == "qwen2.5:3b"


@pytest.fixture
def direct(monkeypatch):
    monkeypatch.setattr(
        graph_module,
        "direct_answer",
        lambda model, query: {
            "answer": "raw answer",
            "generate_ms": 123.4,
            "eval_count": 10,
            "tokens_per_sec": 42.0,
        },
    )


def test_ask_direct_returns_the_answer_and_timings(direct):
    result = graph_module.ask_direct("q", model=TEST_MODEL)

    assert result["answer"] == "raw answer"
    assert result["model"] == TEST_MODEL
    assert result["tokens_per_sec"] == 42.0
    assert result["eval_count"] == 10
    assert isinstance(result["latency_ms"], float)


def test_ask_direct_raises_for_an_unrecognized_model(direct):
    with pytest.raises(ValueError, match="model is required"):
        graph_module.ask_direct("q", model="gpt-4")


def test_ask_direct_never_touches_history(direct, monkeypatch):
    fake_history = FakeHistory()
    monkeypatch.setattr(graph_module, "history", fake_history)

    graph_module.ask_direct("q", model=TEST_MODEL)

    assert fake_history.completed is None
    assert fake_history.failed is None


def test_ask_stream_reports_the_refusal_the_generator_declared(monkeypatch, streaming):
    def refusing_stream(state):
        yield "no context"
        return {"answer": "no context", "citations": [], "refused": True, "confidence": 0.0}

    monkeypatch.setattr(graph_module, "generate_stream", refusing_stream)

    done = list(graph_module.ask_stream("q", model=TEST_MODEL))[-1]

    assert done["refused"] is True


def test_an_answer_with_no_citable_chunks_is_not_reported_as_a_refusal(
    monkeypatch, streaming
):
    def uncitable_stream(state):
        yield "the answer"
        return {
            "answer": "the answer",
            "citations": [],
            "refused": False,
            "confidence": 0.71,
        }

    monkeypatch.setattr(graph_module, "generate_stream", uncitable_stream)

    done = list(graph_module.ask_stream("q", model=TEST_MODEL))[-1]

    assert done["refused"] is False
    assert done["citations"] == []


def test_ask_stream_closing_early_records_the_partial_answer(streaming):
    events = graph_module.ask_stream("what?", model=TEST_MODEL)
    seen = []
    for event in events:
        seen.append(event)
        if event["type"] == "token":
            break
    events.close()

    assert streaming.cancelled == ("entry-1", "the ")
    assert streaming.completed is None


def test_ask_stream_reports_a_failure_as_an_error_event(monkeypatch, streaming):
    def _boom(state):
        raise RuntimeError("ollama unreachable")

    monkeypatch.setattr(graph_module, "retrieve_node", _boom)

    events = list(graph_module.ask_stream("q", model=TEST_MODEL))

    assert events[0] == {"type": "stage", "stage": "retrieve"}
    assert events[-1]["type"] == "error"
    assert "ollama unreachable" in events[-1]["detail"]
    assert streaming.failed == "entry-1"
