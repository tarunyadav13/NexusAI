import re
from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, documents):

        self.documents = documents

        self.tokenized_documents = [
            self._tokenize(doc["content"])
            for doc in documents
        ]

        self.bm25 = (
            BM25Okapi(self.tokenized_documents)
            if self.tokenized_documents
            else None
        )

    @staticmethod
    def _tokenize(text):

        return re.findall(
            r"\b\w+\b",
            text.lower()
        )

    def search(self, query, top_k=10):

        if not query.strip() or not self.documents:
            return []

        tokens = self._tokenize(query)

        scores = self.bm25.get_scores(tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for index in ranked_indices:

            if len(results) >= top_k:
                break

            document = self.documents[index].copy()

            document["bm25_score"] = float(scores[index])

            results.append(document)

        return results