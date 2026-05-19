"""Pydantic models for LLM Wiki rebuild endpoints."""

from typing import Literal

from pydantic import BaseModel, Field


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
