class IngestionError(Exception):
    """
    Base exception for document ingestion.
    """
    pass


class UnsupportedFileTypeError(IngestionError):
    """
    Raised when an unsupported document format is uploaded.
    """
    pass


class DocumentExtractionError(IngestionError):
    """
    Raised when document text extraction fails.
    """
    pass


class EmptyDocumentError(IngestionError):
    """
    Raised when a document contains no extractable text.
    """
    pass


class DuplicateDocumentError(IngestionError):
    """
    Raised when a duplicate document is detected.
    """
    pass


class ChunkingError(IngestionError):
    """
    Raised when document chunking fails.
    """
    pass


class MetadataGenerationError(IngestionError):
    """
    Raised when metadata generation fails.
    """
    pass