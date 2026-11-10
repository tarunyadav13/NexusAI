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

from nexusai.ingestion import IngestionPipeline


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        )
    )


def print_result(result):
    print(
        json.dumps(
            result.model_dump(mode="json"),
            indent=4
        )
    )


def main():

    parser = argparse.ArgumentParser(
        description="NexusAI Multi-Source Ingestion CLI"
    )

    parser.add_argument(
        "--input",
        required=False,
        help="Document file or directory to ingest"
    )

    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Webpage URL to ingest. Can be supplied multiple times."
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200
    )

    args = parser.parse_args()

    if not args.input and not args.url:
        parser.error(
            "Provide either --input, --url, or both."
        )

    configure_logging()

    pipeline = IngestionPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    results = []

    # -----------------------------------------
    # 1. DOCUMENT INGESTION
    # -----------------------------------------

    if args.input:

        input_path = Path(args.input)

        if not input_path.exists():
            parser.error(
                f"Input path does not exist: {input_path}"
            )

        if input_path.is_file():

            files = [input_path]

        else:

            files = [
                path
                for path in input_path.rglob("*")
                if path.is_file()
                and path.suffix.lower() in {
                    ".pdf",
                    ".docx",
                    ".txt"
                }
            ]

        print(
            f"\nFound {len(files)} document(s).\n"
        )

        for file_path in files:

            print(
                f"\nProcessing document: {file_path.name}"
            )

            try:

                result = pipeline.ingest(file_path)

                results.append(
                    result.model_dump(mode="json")
                )

                print_result(result)

            except Exception as exc:

                logging.exception(
                    "Failed to ingest document: %s",
                    file_path
                )

                results.append({
                    "filename": file_path.name,
                    "status": "failed",
                    "error": str(exc)
                })

    # -----------------------------------------
    # 2. URL INGESTION
    # -----------------------------------------

    for url in args.url:

        print(
            f"\nProcessing URL: {url}"
        )

        try:

            result = pipeline.ingest_url(url)

            results.append(
                result.model_dump(mode="json")
            )

            print_result(result)

        except Exception as exc:

            logging.exception(
                "Failed to ingest URL: %s",
                url
            )

            results.append({
                "source": url,
                "status": "failed",
                "error": str(exc)
            })

    # -----------------------------------------
    # 3. SUMMARY
    # -----------------------------------------

    successful = sum(
        result.get("status") == "processed"
        for result in results
    )

    duplicates = sum(
        result.get("status") == "duplicate"
        for result in results
    )

    failed = sum(
        result.get("status") == "failed"
        for result in results
    )

    print("\n========== INGESTION SUMMARY ==========")

    print(f"Total Sources  : {len(results)}")
    print(f"Processed      : {successful}")
    print(f"Duplicates     : {duplicates}")
    print(f"Failed         : {failed}")

    print("=======================================\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()