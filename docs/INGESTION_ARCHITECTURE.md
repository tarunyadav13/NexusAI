# NexusAI - Document Ingestion Architecture

## 1. Overview

NexusAI implements a modular document ingestion pipeline
for processing unstructured documents before embedding
generation and vector storage.

Supported formats:

- PDF
- DOCX
- TXT

## 2. Architecture

Document Upload
    |
Document Loader
    |
Text Extraction
    |
Text Cleaning
    |
SHA-256 Fingerprinting
    |
Duplicate Detection
    |
Metadata Generation
    |
Recursive Character Chunking
    |
Chunk Metadata Enrichment
    |
JSONL Persistence

## 3. Technology Stack

- Python
- PyMuPDF
- python-docx
- Pydantic
- LangChain Text Splitters
- SHA-256
- Pytest

## 4. Chunking Configuration

Initial configuration:

- Chunk size: 1200 characters
- Chunk overlap: 200 characters

The chunking configuration will be evaluated
against alternative configurations.

## 5. Duplicate Detection

SHA-256 fingerprints are generated from
the original document bytes.

Previously processed fingerprints are stored
in a persistent JSON manifest.

## 6. Metadata

Document-level metadata:

- Document ID
- Filename
- Source path
- File type
- SHA-256 fingerprint
- Page count
- Ingestion timestamp

Chunk-level metadata:

- Chunk ID
- Document ID
- Chunk index
- Page numbers
- Filename
- Source information
- File type

## 7. Error Handling

Custom exceptions are implemented for:

- Unsupported formats
- Extraction failures
- Empty documents
- Chunking failures
- Metadata failures

## 8. Future Extensions

- OCR support
- Semantic chunking
- Table extraction
- Image extraction
- Layout-aware parsing
- Distributed ingestion
- Qdrant integration
- Embedding generation