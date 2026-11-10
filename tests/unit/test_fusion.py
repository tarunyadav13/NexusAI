from nexusai.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_fusion():

    dense = [
        {"chunk_id": "1", "content": "A"},
        {"chunk_id": "2", "content": "B"}
    ]

    sparse = [
        {"chunk_id": "2", "content": "B"},
        {"chunk_id": "3", "content": "C"}
    ]

    results = reciprocal_rank_fusion(
        [dense, sparse],
        top_k=3
    )

    assert len(results) == 3

    assert results[0]["chunk_id"] == "2"


def test_empty_fusion():

    assert reciprocal_rank_fusion([]) == []