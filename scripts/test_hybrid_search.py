import argparse
import logging
import time

from nexusai.retrieval.hybrid_retriever import HybridRetriever


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        required=True
    )

    parser.add_argument(
        "--without-reranker",
        action="store_true"
    )

    args = parser.parse_args()

    retriever = HybridRetriever(
        use_reranker=not args.without_reranker
    )

    try:

        start = time.perf_counter()

        response = retriever.search(args.query)

        latency = (
            time.perf_counter() - start
        ) * 1000

        print("\n========== NEXUSAI HYBRID SEARCH ==========\n")

        print("Query:", response["query"])

        print("Dense results:", response["dense_count"])

        print("Sparse results:", response["sparse_count"])

        print("Fused results:", response["fused_count"])

        print("Final results:", len(response["results"]))

        print(f"Latency: {latency:.2f} ms")

        print("\n========== FINAL RESULTS ==========\n")

        for index, result in enumerate(
            response["results"],
            1
        ):

            print(f"RESULT {index}")

            print("Source:", result["source"])

            print("Content:", result["content"])

            if "rrf_score" in result:
                print("RRF Score:", result["rrf_score"])

            if "reranker_score" in result:
                print(
                    "Reranker Score:",
                    result["reranker_score"]
                )

            print("-" * 60)

    finally:

        retriever.close()


if __name__ == "__main__":
    main()