from nexusai.ingestion.chunker import DocumentChunker

from nexusai.ingestion.schemas import ExtractedPage

import pytest


def test_chunk_generation():

    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=20
    )

    text = "Artificial intelligence " * 100

    pages = [
        ExtractedPage(
            page_number=1,
            text=text
        )
    ]

    chunks = chunker.chunk_pages(
        pages,
        document_id="test123"
    )

    assert len(chunks) > 1

    assert all(
        len(chunk.text) <= 100
        for chunk in chunks
    )


def test_chunk_metadata():

    chunker = DocumentChunker()

    pages = [
        ExtractedPage(
            page_number=5,
            text="NexusAI is a RAG system."
        )
    ]

    chunks = chunker.chunk_pages(
        pages,
        "document123"
    )

    assert chunks[0].page_numbers == [5]

    assert chunks[0].document_id == "document123"


def test_invalid_chunk_configuration():

    with pytest.raises(ValueError):

        DocumentChunker(
            chunk_size=100,
            chunk_overlap=100
        )


def test_empty_page():

    chunker = DocumentChunker()

    pages = [
        ExtractedPage(
            page_number=1,
            text=""
        )
    ]

    chunks = chunker.chunk_pages(
        pages,
        "document123"
    )

    assert chunks == []