import logging

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from nexusai.ingestion.schemas import (
    ExtractedPage,
    DocumentChunk
)

from nexusai.ingestion.exceptions import ChunkingError


logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Splits extracted document pages into overlapping chunks.
    """

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200
    ):

        if chunk_size <= 0:
            raise ValueError(
                "Chunk size must be positive."
            )

        if chunk_overlap < 0:
            raise ValueError(
                "Chunk overlap cannot be negative."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "Chunk overlap must be smaller than chunk size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ],
            length_function=len,
            is_separator_regex=False
        )

    def chunk_pages(
        self,
        pages: list[ExtractedPage],
        document_id: str
    ) -> list[DocumentChunk]:

        chunks = []

        chunk_index = 0

        try:

            for page in pages:

                text = page.text.strip()

                if not text:
                    continue

                page_chunks = self.splitter.split_text(
                    text
                )

                for chunk_text in page_chunks:

                    if not chunk_text.strip():
                        continue

                    chunk = DocumentChunk(
                        chunk_id=f"{document_id}_{chunk_index}",
                        document_id=document_id,
                        chunk_index=chunk_index,
                        text=chunk_text,
                        page_numbers=[
                            page.page_number
                        ],
                        metadata={
                            "chunk_size": len(chunk_text),
                            "chunk_overlap": self.chunk_overlap,
                            "page_number": page.page_number
                        }
                    )

                    chunks.append(chunk)

                    chunk_index += 1

            logger.info(
                "Generated %d chunks for document %s",
                len(chunks),
                document_id
            )

            return chunks

        except Exception as exc:

            logger.exception(
                "Chunking failed for document %s",
                document_id
            )

            raise ChunkingError(
                f"Unable to chunk document: {exc}"
            ) from exc