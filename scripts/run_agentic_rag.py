import argparse
import json
import logging
import sys

from pathlib import Path


# Make the src directory accessible
PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

from nexusai.agentic.graph import build_agentic_graph


def main():

    parser = argparse.ArgumentParser(
        description="NexusAI Agentic RAG"
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Question to ask NexusAI"
    )

    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable external web search"
    )

    args = parser.parse_args()

    retriever = None

    try:

        print("\nInitializing NexusAI Agentic RAG...\n")

        graph, retriever = build_agentic_graph()

        # -----------------------------------------
        # INITIAL AGENT STATE
        # -----------------------------------------

        initial_state = {
            "original_query": args.query,
            "rewritten_query": args.query,
            "current_query": args.query,

            "documents": [],

            "relevance_score": 0.0,
            "evidence_sufficient": False,

            "retry_count": 0,
            "max_retries": 2,

            "answer": "",
            "citations": [],

            "web_enabled": not args.no_web,
            "web_results": [],

            "error": None
        }

        print(
            "Web Search:",
            "Disabled" if args.no_web else "Enabled"
        )

        # -----------------------------------------
        # RUN AGENTIC GRAPH
        # -----------------------------------------

        result = graph.invoke(initial_state)

        # -----------------------------------------
        # DISPLAY RESULT
        # -----------------------------------------

        print("\n" + "=" * 60)
        print("NEXUSAI AGENTIC RAG RESPONSE")
        print("=" * 60)

        print("\nQuestion:")
        print(args.query)

        print("\nAnswer:")
        print(
            result.get(
                "answer",
                "No answer generated."
            )
        )

        print("\nCitations:")

        print(
            json.dumps(
                result.get("citations", []),
                indent=2
            )
        )

        print("\nAgentic RAG Execution:")

        print(
            "Final query:",
            result.get("current_query")
        )

        print(
            "Retry count:",
            result.get("retry_count", 0)
        )

        print(
            "Evidence score:",
            result.get("relevance_score", 0)
        )

        print(
            "Evidence sufficient:",
            result.get("evidence_sufficient", False)
        )

        print(
            "Web search enabled:",
            result.get("web_enabled", False)
        )

        print(
            "Web results:",
            len(result.get("web_results", []))
        )

        print("\n" + "=" * 60)

    except Exception:

        logging.exception(
            "Agentic RAG execution failed."
        )

        raise

    finally:

        if retriever is not None:
            retriever.close()


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s"
    )

    main()