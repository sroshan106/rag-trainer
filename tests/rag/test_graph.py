from src.rag.graph import MAX_RETRIES, build_graph, should_retry


def test_should_retry_retries_on_empty_graded_docs_below_limit():
    state = {"graded_docs": [], "retry_count": 0}

    assert should_retry(state) == "retrieve"


def test_should_retry_generates_when_docs_present():
    state = {"graded_docs": ["doc"], "retry_count": 0}

    assert should_retry(state) == "generate"


def test_should_retry_generates_after_max_retries():
    state = {"graded_docs": [], "retry_count": MAX_RETRIES}

    assert should_retry(state) == "generate"


def test_build_graph_compiles():
    graph = build_graph()

    assert graph is not None
