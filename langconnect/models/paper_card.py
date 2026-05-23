from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class PaperCardExtractionQuality(BaseModel):
    has_title: bool = False
    has_abstract: bool = False
    abstract_chars: int = 0
    abstract_sentence_like_spans: int = 0
    warnings: list[str] = Field(default_factory=list)


class PaperCardV0(BaseModel):
    schema_version: Literal["paper-card/v0"] = "paper-card/v0"
    collection_id: str
    source: str
    filename: str
    source_path: str | None = None
    content_hash: str
    parser: str
    parser_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    title: str | None = None
    abstract: str | None = None
    extraction_quality: PaperCardExtractionQuality
