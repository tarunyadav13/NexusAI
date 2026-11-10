from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict


class ExtractedPage(BaseModel):
    page_number: int = Field(ge=1)
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoadedDocument(BaseModel):
    filename: str
    source_path: str
    file_type: Literal["pdf", "docx", "txt", "url"]
    pages: list[ExtractedPage]

    @property
    def page_count(self) -> int:
        return len(self.pages)


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    source_path: str
    file_type: str
    sha256: str
    page_count: int

    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    page_numbers: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionResult(BaseModel):
    document_id: str | None = None
    source_path: str

    status: Literal[
        "processed",
        "duplicate",
        "failed"
    ]

    page_count: int = 0
    chunk_count: int = 0
    processing_time_seconds: float = 0.0
    fingerprint: str | None = None
    output_path: str | None = None
    error: str | None = None