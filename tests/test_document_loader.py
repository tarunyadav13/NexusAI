import fitz

from docx import Document

from nexusai.ingestion.document_loader import DocumentLoader

from nexusai.ingestion.exceptions import (
    UnsupportedFileTypeError,
    EmptyDocumentError
)

import pytest


def test_load_txt(tmp_path):

    file = tmp_path / "sample.txt"

    file.write_text(
        "NexusAI is an advanced RAG system.",
        encoding="utf-8"
    )

    loader = DocumentLoader()

    result = loader.load(file)

    assert result.file_type == "txt"

    assert len(result.pages) == 1

    assert "NexusAI" in result.pages[0].text


def test_load_pdf(tmp_path):

    file = tmp_path / "sample.pdf"

    pdf = fitz.open()

    page = pdf.new_page()

    page.insert_text(
        (50, 50),
        "NexusAI PDF testing"
    )

    pdf.save(file)

    pdf.close()

    loader = DocumentLoader()

    result = loader.load(file)

    assert result.file_type == "pdf"

    assert len(result.pages) == 1

    assert "NexusAI" in result.pages[0].text


def test_load_docx(tmp_path):

    file = tmp_path / "sample.docx"

    document = Document()

    document.add_paragraph(
        "NexusAI DOCX testing"
    )

    document.save(file)

    loader = DocumentLoader()

    result = loader.load(file)

    assert result.file_type == "docx"

    assert "NexusAI" in result.pages[0].text


def test_unsupported_file(tmp_path):

    file = tmp_path / "sample.csv"

    file.write_text("test")

    loader = DocumentLoader()

    with pytest.raises(
        UnsupportedFileTypeError
    ):

        loader.load(file)


def test_empty_document(tmp_path):

    file = tmp_path / "empty.txt"

    file.write_text("")

    loader = DocumentLoader()

    with pytest.raises(
        EmptyDocumentError
    ):

        loader.load(file)