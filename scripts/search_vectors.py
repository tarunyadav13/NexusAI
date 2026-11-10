import argparse
import logging

from nexusai.embeddings.generator import EmbeddingGenerator
from nexusai.vectorstore.qdrant_store import QdrantStore


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        required=True,
        help="Natural language search query"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5
    )

    args = parser.parse_args()

    embedder = EmbeddingGenerator()

    store = QdrantStore()

    try:

        query_vector = embedder.embed_query(
            args.query
        )

        results = store.search(

            query_vector=query_vector,

            limit=args.top_k
        )

        print("\n========== SEMANTIC SEARCH RESULTS ==========\n")

        for index, result in enumerate(results, 1):

            print(f"RESULT {index}")

            print(
                f"Similarity Score: {result['score']:.4f}"
            )

            print(
                f"Source: {result['metadata'].get('source')}"
            )

            print(
                f"Document ID: {result['metadata'].get('document_id')}"
            )

            print("\nContent:")

            print(result["text"])

            print("\n" + "-" * 55)

        print(
            f"\nTotal results: {len(results)}"
        )

    finally:

        embedder.close()

        store.close()


if __name__ == "__main__":

    main()