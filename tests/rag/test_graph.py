from src.rag.graph import build_graph


def test_build_graph_compiles():
    graph = build_graph()

    assert graph is not None


def test_graph_has_no_retry_edge_back_into_retrieve():
    # Retrieval is deterministic and rank-ordered, so a retry edge could
    # never change the grader's verdict. Guard against it creeping back.
    edges = build_graph().get_graph().edges

    assert not [e for e in edges if e.source == "grade" and e.target == "retrieve"]
