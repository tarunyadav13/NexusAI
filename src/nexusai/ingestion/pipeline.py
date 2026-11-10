import json
import time
import hashlib
import logging

from pathlib import Path
from urllib.parse import urlparse

from nexusai.ingestion.document_loader import DocumentLoader
from nexusai.ingestion.url_loader import URLLoader

from nexusai.ingestion.text_cleaner import TextCleaner
from nexusai.ingestion.chunker import DocumentChunker
from nexusai.ingestion.metadata import MetadataBuilder
from nexusai.ingestion.deduplication import DocumentDeduplicator

from nexusai.ingestion.schemas import IngestionResult
from nexusai.ingestion.exceptions import IngestionError


logger = logging.getLogger(__name__)


class IngestionPipeline:

    def __init__(
        self,
        chunk_size=1200,
        chunk_overlap=200,
        output_dir="data/processed",
        manifest_path=None
    ):

        self.loader = DocumentLoader()
        self.url_loader = URLLoader()

        self.cleaner = TextCleaner()

        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.metadata_builder = MetadataBuilder()

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        if manifest_path is None:
            manifest_path = (
                self.output_dir / "ingestion_manifest.json"
            )

        self.deduplicator = DocumentDeduplicator(
            manifest_path=manifest_path
        )

    def ingest(self, file_path):

        start = time.perf_counter()

        path = Path(file_path)

        try:

            if not path.exists() or not path.is_file():
                raise FileNotFoundError(
                    f"Invalid document path: {path}"
                )

            fingerprint = self.deduplicator.calculate_hash(path)

            document = self.loader.load(path)

            return self._process_document(
                document=document,
                fingerprint=fingerprint,
                start=start
            )

        except Exception as exc:

            logger.exception("File ingestion failed")

            return self._failure(
                str(path),
                start,
                str(exc)
            )

    def ingest_url(self, url):

        start = time.perf_counter()

        try:

            document = self.url_loader.load(url)

            fingerprint = hashlib.sha256(
                url.strip().encode("utf-8")
            ).hexdigest()

            return self._process_document(
                document=document,
                fingerprint=fingerprint,
                start=start
            )

        except Exception as exc:

            logger.exception("URL ingestion failed")

            return self._failure(
                url,
                start,
                str(exc)
            )

    def _process_document(
        self,
        document,
        fingerprint,
        start
    ):

        existing = self.deduplicator.find_duplicate(
            fingerprint
        )

        if existing:

            return IngestionResult(
                document_id=existing["document_id"],
                source_path=document.source_path,
                status="duplicate",
                fingerprint=fingerprint,
                output_path=existing.get("output_path"),
                processing_time_seconds=round(
                    time.perf_counter() - start,
                    4
                )
            )

        for page in document.pages:

            page.text = self.cleaner.clean(
                page.text
            )

        document.pages = [
            page for page in document.pages
            if page.text
        ]

        if not document.pages:
            raise IngestionError(
                "No usable text found."
            )

        document_metadata = (
            self.metadata_builder.create_document_metadata(
                document=document,
                fingerprint=fingerprint
            )
        )

        document_id = document_metadata.document_id

        chunks = self.chunker.chunk_pages(
            document.pages,
            document_id
        )

        chunks = self.metadata_builder.enrich_chunks(
            chunks,
            document_metadata
        )

        source_type = (
            "webpage"
            if document.file_type == "url"
            else "document"
        )

        for chunk in chunks:

            chunk.metadata.update({
                "source_type": source_type,
                "source": document.source_path,
                "source_url": (
                    document.source_path
                    if source_type == "webpage"
                    else None
                ),
                "title": document.pages[0].metadata.get(
                    "title",
                    document.filename
                )
            })

        output_path = (
            self.output_dir /
            f"{document_id}.jsonl"
        )

        self._save_chunks(
            chunks,
            output_path
        )

        self.deduplicator.register_document(
            fingerprint=fingerprint,
            document_id=document_id,
            filename=document.filename,
            source_path=document.source_path,
            output_path=str(output_path.resolve())
        )

        return IngestionResult(
            document_id=document_id,
            source_path=document.source_path,
            status="processed",
            page_count=document.page_count,
            chunk_count=len(chunks),
            fingerprint=fingerprint,
            output_path=str(output_path.resolve()),
            processing_time_seconds=round(
                time.perf_counter() - start,
                4
            )
        )

    @staticmethod
    def _save_chunks(chunks, output_path):

        temporary_path = output_path.with_suffix(".tmp")

        try:

            with temporary_path.open(
                "w",
                encoding="utf-8"
            ) as file:

                for chunk in chunks:

                    file.write(
                        json.dumps(
                            chunk.model_dump(mode="json"),
                            ensure_ascii=False
                        )
                    )

                    file.write("\n")

            temporary_path.replace(output_path)

        except Exception:

            temporary_path.unlink(missing_ok=True)

            raise

    @staticmethod
    def _failure(source, start, error):

        return IngestionResult(
            source_path=str(source),
            status="failed",
            error=error,
            processing_time_seconds=round(
                time.perf_counter() - start,
                4
            )
        )