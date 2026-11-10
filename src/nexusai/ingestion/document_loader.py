import logging
from pathlib import Path

import pymupdf
from docx import Document as DocxDocument

from nexusai.ingestion.schemas import (
    ExtractedPage,
    LoadedDocument
)

from nexusai.ingestion.exceptions import (
    UnsupportedFileTypeError,
    DocumentExtractionError,
    EmptyDocumentError
)


logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Extracts text from supported document formats.
    """

    SUPPORTED_FORMATS = {
        ".pdf",
        ".docx",
        ".txt"
    }

    def load(self, file_path: str | Path) -> LoadedDocument:

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {path}"
            )

        if not path.is_file():
            raise DocumentExtractionError(
                f"Path is not a file: {path}"
            )

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_FORMATS:
            raise UnsupportedFileTypeError(
                f"Unsupported format: {extension}"
            )

        logger.info(
            "Loading document: %s",
            path.name
        )

        try:

            if extension == ".pdf":
                pages = self._load_pdf(path)

            elif extension == ".docx":
                pages = self._load_docx(path)

            else:
                pages = self._load_txt(path)

            if not pages or not any(
                page.text.strip() for page in pages
            ):
                raise EmptyDocumentError(
                    f"No extractable text found: {path.name}. "
                    "Scanned PDFs may require OCR."
                )

            return LoadedDocument(
                filename=path.name,
                source_path=str(path.resolve()),
                file_type=extension[1:],
                pages=pages
            )

        except (
            UnsupportedFileTypeError,
            EmptyDocumentError,
            DocumentExtractionError
        ):
            raise

        except Exception as exc:

            logger.exception(
                "Document extraction failed: %s",
                path.name
            )

            raise DocumentExtractionError(
                f"Failed to extract {path.name}: {exc}"
            ) from exc

    def _load_pdf(self, path: Path) -> list[ExtractedPage]:

        pages = []

        with pymupdf.open(path) as pdf:

            for index, page in enumerate(pdf):

                text = page.get_text("text")

                pages.append(
                    ExtractedPage(
                        page_number=index + 1,
                        text=text,
                        metadata={
                            "source_format": "pdf"
                        }
                    )
                )

        return pages

    def _load_docx(self, path: Path) -> list[ExtractedPage]:

        document = DocxDocument(path)

        content = []

        for paragraph in document.paragraphs:

            if paragraph.text.strip():
                content.append(paragraph.text)

        for table in document.tables:

            for row in table.rows:

                row_text = " | ".join(
                    cell.text.strip()
                    for cell in row.cells
                )

                if row_text.strip():
                    content.append(row_text)

        text = "\n".join(content)

        return [
            ExtractedPage(
                page_number=1,
                text=text,
                metadata={
                    "source_format": "docx"
                }
            )
        ]

    def _load_txt(self, path: Path) -> list[ExtractedPage]:

        try:

            text = path.read_text(
                encoding="utf-8-sig"
            )

        except UnicodeDecodeError:

            text = path.read_text(
                encoding="cp1252"
            )

        return [
            ExtractedPage(
                page_number=1,
                text=text,
                metadata={
                    "source_format": "txt"
                }
            )
        ]