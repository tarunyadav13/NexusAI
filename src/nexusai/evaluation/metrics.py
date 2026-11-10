import math


def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]

    if k <= 0:
        raise ValueError("k must be greater than zero")

    relevant_set = set(relevant)

    hits = sum(
        1 for item in retrieved_k
        if item in relevant_set
    )

    return hits / k


def recall_at_k(retrieved, relevant, k):
    relevant_set = set(relevant)

    if not relevant_set:
        return 0.0

    retrieved_k = retrieved[:k]

    hits = sum(
        1 for item in retrieved_k
        if item in relevant_set
    )

    return hits / len(relevant_set)


def reciprocal_rank(retrieved, relevant):
    relevant_set = set(relevant)

    for rank, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(retrieved, relevant, k):
    relevant_set = set(relevant)

    if k <= 0:
        raise ValueError("k must be greater than zero")

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(retrieved[:k], start=1)
        if item in relevant_set
    )

    ideal_hits = min(len(relevant_set), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg else 0.0