import json
import logging

from pathlib import Path

from nexusai.embeddings.generator import EmbeddingGenerator
from nexusai.vectorstore.qdrant_store import QdrantStore


ROOT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT_DIR / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def extract_chunks():

    chunks = []

    files = list(PROCESSED_DIR.glob("*.jsonl"))

    if not files:
        raise FileNotFoundError(
            "No processed JSONL files found."
        )

    for file_path in files:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            for line_number, line in enumerate(file):

                if not line.strip():
                    continue

                record = json.loads(line)

                records = (
                    record.get("chunks", [])
                    if isinstance(record.get("chunks"), list)
                    else [record]
                )

                for item in records:

                    if not isinstance(item, dict):
                        continue

                    metadata = item.get("metadata", {})

                    if not isinstance(metadata, dict):
                        metadata = {}

                    text = (
                        item.get("text")
                        or item.get("content")
                        or item.get("chunk_text")
                        or item.get("page_content")
                    )

                    if not text:
                        continue

                    document_id = (
                        item.get("document_id")
                        or file_path.stem
                    )

                    chunk_index = (
                        item.get("chunk_index")
                        if item.get("chunk_index") is not None
                        else line_number
                    )

                    chunk_metadata = {
                        **metadata,

                        "document_id": str(document_id),

                        "chunk_index": str(chunk_index),

                        "source": metadata.get(
                            "source",
                            metadata.get(
                                "source_path",
                                file_path.name
                            )
                        ),

                        "source_type": metadata.get(
                            "source_type",
                            "document"
                        ),

                        "source_url": metadata.get(
                            "source_url"
                        ),

                        "title": metadata.get(
                            "title",
                            ""
                        ),

                        "page_numbers": item.get(
                            "page_numbers",
                            metadata.get("page_numbers", [])
                        )
                    }

                    chunks.append({
                        "text": text,
                        "metadata": chunk_metadata
                    })

    return chunks


def main():

    logging.info("Starting vector indexing")

    chunks = extract_chunks()

    if not chunks:
        raise ValueError("No valid chunks found.")

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    metadata = [
        chunk["metadata"]
        for chunk in chunks
    ]

    embedding_generator = EmbeddingGenerator()

    vector_store = QdrantStore()

    try:

        embeddings = embedding_generator.embed_documents(
            texts
        )

        stored = vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadata=metadata
        )

        print("\n========== INDEXING SUMMARY ==========")
        print(f"Chunks extracted : {len(chunks)}")
        print(f"Embeddings created: {len(embeddings)}")
        print(f"Vectors stored   : {stored}")
        print("Vector database  : Qdrant")
        print("======================================")

    finally:

        embedding_generator.close()
        vector_store.close()


if __name__ == "__main__":
    main()