from langchain_core.documents import Document

from src.rag import nodes


class FakeVectorstore:
    def __init__(self, docs):
        self.docs = docs
        self.last_query = None
        self.last_k = None

    def similarity_search(self, query, k):
        self.last_query = query
        self.last_k = k
        return self.docs


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


def test_retrieve_node_populates_docs_and_increments_retry(monkeypatch):
    docs = [Document(page_content="a", metadata={})]
    fake_store = FakeVectorstore(docs)
    monkeypatch.setattr(nodes, "_get_vectorstore", lambda: fake_store)

    result = nodes.retrieve_node({"query": "q", "retry_count": 1})

    assert result["retrieved_docs"] == docs
    assert result["retry_count"] == 2
    assert fake_store.last_query == "q"
    assert fake_store.last_k == nodes.RETRIEVE_K


def test_retrieve_node_defaults_retry_count(monkeypatch):
    monkeypatch.setattr(nodes, "_get_vectorstore", lambda: FakeVectorstore([]))

    result = nodes.retrieve_node({"query": "q"})

    assert result["retry_count"] == 1


def test_grade_node_drops_blank_chunks():
    docs = [
        Document(page_content="keep me", metadata={}),
        Document(page_content="   ", metadata={}),
        Document(page_content="", metadata={}),
    ]

    result = nodes.grade_node({"retrieved_docs": docs})

    assert len(result["graded_docs"]) == 1
    assert result["graded_docs"][0].page_content == "keep me"


def test_generate_node_returns_fallback_when_no_graded_docs():
    result = nodes.generate_node({"query": "q", "graded_docs": []})

    assert "don't have enough context" in result["answer"]


def test_generate_node_calls_llm_with_formatted_context(monkeypatch):
    fake_llm = FakeLLM("the answer")
    monkeypatch.setattr(nodes, "_get_llm", lambda: fake_llm)
    docs = [Document(page_content="fact one", metadata={"source": "a.txt"})]

    result = nodes.generate_node({"query": "what?", "graded_docs": docs})

    assert result["answer"] == "the answer"
    assert "fact one" in fake_llm.last_prompt
    assert "what?" in fake_llm.last_prompt
