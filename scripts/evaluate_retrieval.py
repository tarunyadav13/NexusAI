import json
import logging
import time
from pathlib import Path

from nexusai.retrieval.hybrid_retriever import HybridRetriever

from nexusai.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]

EVALUATION_DIR = (
    ROOT_DIR / "src" / "nexusai" / "evaluation"
)

DATASET_PATH = (
    EVALUATION_DIR / "datasets" / "retrieval_eval.json"
)

RESULTS_DIR = EVALUATION_DIR / "results"

K = 5


def normalize_source(source):
    return str(source).replace("\\", "/").strip().lower()


def source_matches(retrieved_source, relevant_source):
    retrieved = normalize_source(retrieved_source)
    relevant = normalize_source(relevant_source)

    return (
        retrieved == relevant
        or retrieved.endswith("/" + relevant)
    )


def evaluate_query(retriever, test_case):

    query = test_case["query"]
    relevant_sources = test_case["relevant_sources"]

    start_time = time.perf_counter()

    response = retriever.search(query)

    latency = time.perf_counter() - start_time

    results = response.get("results", [])

    # Deduplicate retrieved documents by source.
    retrieved_sources = []
    seen_sources = set()

    for result in results:

        source = result.get("source", "")
        normalized = normalize_source(source)

        if normalized and normalized not in seen_sources:
            seen_sources.add(normalized)
            retrieved_sources.append(source)

    # Match retrieved sources against ground-truth sources.
    retrieved_ids = []

    for source in retrieved_sources:

        matching_source = next(
            (
                relevant
                for relevant in relevant_sources
                if source_matches(source, relevant)
            ),
            None
        )

        retrieved_ids.append(matching_source or source)

    relevant_ids = list(set(relevant_sources))

    # Evaluate only the number of unique results actually retrieved.
    evaluation_k = min(K, len(retrieved_ids))

    if evaluation_k == 0:
        precision = recall = mrr = ndcg = 0.0

    else:
        precision = precision_at_k(
            retrieved_ids, relevant_ids, evaluation_k
        )

        recall = recall_at_k(
            retrieved_ids, relevant_ids, evaluation_k
        )

        mrr = reciprocal_rank(
            retrieved_ids, relevant_ids
        )

        ndcg = ndcg_at_k(
            retrieved_ids, relevant_ids, evaluation_k
        )

    return {
        "query": query,
        "precision_at_5": round(precision, 4),
        "recall_at_5": round(recall, 4),
        "mrr": round(mrr, 4),
        "ndcg_at_5": round(ndcg, 4),
        "latency_seconds": round(latency, 4),
        "retrieved_count": len(results),
        "unique_sources_retrieved": len(retrieved_sources)
    }

def main():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    retriever = HybridRetriever(
        final_top_k=K,
        use_reranker=True
    )

    evaluation_results = []

    try:

        for test_case in dataset:

            logger.info(
                "Evaluating query: %s",
                test_case["query"]
            )

            result = evaluate_query(retriever, test_case)

            evaluation_results.append(result)

            print(json.dumps(result, indent=2))

    finally:
        retriever.close()

    metric_names = [
        "precision_at_5",
        "recall_at_5",
        "mrr",
        "ndcg_at_5",
        "latency_seconds"
    ]

    summary = {}

    for metric in metric_names:

        summary[metric] = round(
            sum(item[metric] for item in evaluation_results)
            / len(evaluation_results),
            4
        ) if evaluation_results else 0.0

    output = {
        "total_queries": len(evaluation_results),
        "summary": summary,
        "individual_results": evaluation_results
    }

    output_path = RESULTS_DIR / "retrieval_results.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print("\n" + "=" * 55)
    print("NEXUSAI RETRIEVAL EVALUATION")
    print("=" * 55)

    print(json.dumps(summary, indent=4))

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()