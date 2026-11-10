import json
import logging
import statistics
import time

from pathlib import Path

from nexusai.retrieval.hybrid_retriever import HybridRetriever

logging.basicConfig(level=logging.INFO)

ROOT_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    ROOT_DIR / "src/nexusai/evaluation/datasets/retrieval_eval.json"
)

RESULTS_DIR = (
    ROOT_DIR / "src/nexusai/evaluation/results"
)


def main():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    retriever = HybridRetriever(
        final_top_k=5,
        use_reranker=True
    )

    latencies = []

    results = []

    try:

        for case in dataset:

            query = case["query"]

            start = time.perf_counter()

            response = retriever.search(query)

            latency = time.perf_counter() - start

            latencies.append(latency)

            results.append({
                "query": query,
                "latency_seconds": round(latency, 4),
                "results_returned": len(
                    response.get("results", [])
                )
            })

    finally:

        retriever.close()

    summary = {
        "total_queries": len(latencies),
        "mean_latency_seconds": statistics.mean(latencies),
        "median_latency_seconds": statistics.median(latencies),
        "min_latency_seconds": min(latencies),
        "max_latency_seconds": max(latencies)
    }

    output = {
        "summary": summary,
        "individual_results": results
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "benchmark_results.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print("\nNEXUSAI PERFORMANCE BENCHMARK")

    print(json.dumps(summary, indent=4))

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()