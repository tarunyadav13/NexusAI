from datetime import datetime, timezone
from uuid import uuid4

from nexusai.ingestion.schemas import (
    DocumentMetadata,
    DocumentChunk,
    LoadedDocument
)

from nexusai.ingestion.exceptions import (
    MetadataGenerationError
)


class MetadataBuilder:
    """
    Generates standardized document and chunk metadata.
    """

    def create_document_metadata(
        self,
        document: LoadedDocument,
        fingerprint: str
    ) -> DocumentMetadata:

        try:

            return DocumentMetadata(
                document_id=str(uuid4()),
                filename=document.filename,
                source_path=document.source_path,
                file_type=document.file_type,
                sha256=fingerprint,
                page_count=document.page_count,
                ingested_at=datetime.now(
                    timezone.utc
                )
            )

        except Exception as exc:

            raise MetadataGenerationError(
                f"Document metadata generation failed: {exc}"
            ) from exc

    def enrich_chunks(
        self,
        chunks: list[DocumentChunk],
        document_metadata: DocumentMetadata
    ) -> list[DocumentChunk]:

        for chunk in chunks:

            chunk.metadata.update({

                "filename": document_metadata.filename,

                "source_path": document_metadata.source_path,

                "file_type": document_metadata.file_type,

                "sha256": document_metadata.sha256,

                "ingested_at": document_metadata.ingested_at.isoformat(),

                "page_numbers": chunk.page_numbers

            })

        return chunks