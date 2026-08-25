from app.evaluation.metrics import hit_at_k, recall_at_k


def test_retrieval_metrics():
    assert hit_at_k(['a', 'b'], ['b']) == 1.0
    assert recall_at_k(['a', 'b'], ['b', 'c']) == 0.5
