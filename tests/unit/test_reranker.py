from unittest.mock import MagicMock

from nexusai.retrieval.reranker import BGEReranker


def test_reranking_order():

    reranker = BGEReranker.__new__(
        BGEReranker
    )

    reranker.model = MagicMock()

    reranker.model.predict.return_value = [
        0.2,
        0.9
    ]

    documents = [
        {
            "chunk_id": "1",
            "content": "Irrelevant content"
        },
        {
            "chunk_id": "2",
            "content": "Relevant RAG content"
        }
    ]

    results = reranker.rerank(
        "Explain RAG",
        documents
    )

    assert results[0]["chunk_id"] == "2"