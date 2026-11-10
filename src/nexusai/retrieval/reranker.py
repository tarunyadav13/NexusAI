from sentence_transformers import CrossEncoder


class BGEReranker:

    def __init__(
        self,
        model_name="BAAI/bge-reranker-base"
    ):

        self.model = CrossEncoder(
            model_name,
            max_length=512
        )

    def rerank(
        self,
        query,
        documents,
        top_k=5
    ):

        if not query.strip() or not documents:
            return []

        pairs = [
            [query, doc["content"]]
            for doc in documents
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False
        )

        ranked = []

        for document, score in zip(documents, scores):

            result = document.copy()

            result["reranker_score"] = float(score)

            ranked.append(result)

        ranked.sort(
            key=lambda x: x["reranker_score"],
            reverse=True
        )

        return ranked[:top_k]