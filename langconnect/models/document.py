from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

class DocumentCreate(BaseModel):
    content: str | None = None
    metadata: dict[str, Any] | None = None


class DocumentUpdate(BaseModel):
    content: str | None = None
    metadata: dict[str, Any] | None = None


class DocumentResponse(BaseModel):
    id: str
    collection_id: str
    content: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: str | None = None
    updated_at: str | None = None


class SearchQuery(BaseModel):
    query: str
    limit: int | None = 10
    filter: dict[str, Any] | None = None
    search_type: Literal["semantic", "keyword", "hybrid"] = "semantic"


class SearchResult(BaseModel):
    id: str
    page_content: str
    metadata: dict[str, Any] | None = None
    score: float

class DocumentDelete(BaseModel):
    document_ids: Optional[list[str]] = Field(None, description="List of document IDs to delete.")
    file_ids: Optional[list[str]] = Field(None, description="List of file IDs to delete all associated documents.")
