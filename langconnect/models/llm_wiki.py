"""Pydantic models for LLM Wiki endpoints."""

from typing import Literal

from pydantic import BaseModel, Field


class LLMWikiManifestItem(BaseModel):
    """Public manifest metadata for a generated wiki page."""

    type: Literal["source", "concept"]
    title: str
    path: str
    slug: str
    id: str | None = None
    file_id: str | None = None
    source: str | None = None
    chunk_count: int | None = None
    reference_count: int | None = None


class LLMWikiIndexResponse(BaseModel):
    """Response body for a collection's generated LLM Wiki index."""

    collection_id: str
    status: Literal["available"] = "available"
    generated_at: str | None = None
    index_markdown: str
    sources: list[LLMWikiManifestItem]
    concepts: list[LLMWikiManifestItem]


class LLMWikiPageResponse(BaseModel):
    """Response body for one generated source or concept wiki page."""

    collection_id: str
    section: Literal["sources", "concepts"]
    slug: str
    title: str
    path: str
    markdown: str


class LLMWikiRebuildRequest(BaseModel):
    """Request body for manual collection LLM Wiki rebuilds."""

    llm_provider: str | None = Field(
        None,
        description="LLM provider override: openai, google, or ollama",
    )
    llm_model: str | None = Field(None, description="LLM model override")
    llm_temperature: float | None = Field(
        None,
        ge=0,
        le=2,
        description="LLM temperature override",
    )


class LLMWikiRebuildResponse(BaseModel):
    """Response body for successful collection LLM Wiki rebuilds."""

    collection_id: str
    status: Literal["rebuilt"] = "rebuilt"
    source_page_count: int
    concept_page_count: int
    page_count: int
    chunk_count: int
    pack_path: str
