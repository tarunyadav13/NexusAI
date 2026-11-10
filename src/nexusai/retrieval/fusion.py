def reciprocal_rank_fusion(
    result_lists,
    k=60,
    top_k=10
):

    fused_scores = {}
    documents = {}

    for result_list in result_lists:

        for rank, document in enumerate(result_list):

            doc_id = document["chunk_id"]

            score = 1.0 / (k + rank + 1)

            fused_scores[doc_id] = (
                fused_scores.get(doc_id, 0.0) + score
            )

            documents[doc_id] = document

    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True
    )

    results = []

    for doc_id in ranked_ids[:top_k]:

        result = documents[doc_id].copy()

        result["rrf_score"] = fused_scores[doc_id]

        results.append(result)

    return results