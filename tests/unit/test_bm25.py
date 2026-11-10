from nexusai.retrieval.bm25_retriever import BM25Retriever


def test_bm25_search():

    docs = [
        {
            "chunk_id": "1",
            "content": "Python programming language"
        },
        {
            "chunk_id": "2",
            "content": "Machine learning and neural networks"
        }
    ]

    retriever = BM25Retriever(docs)

    results = retriever.search(
        "Python programming",
        top_k=1
    )

    assert len(results) == 1

    assert results[0]["chunk_id"] == "1"


def test_empty_search():

    retriever = BM25Retriever([])

    assert retriever.search("Python") == []