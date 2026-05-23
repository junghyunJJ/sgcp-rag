"""Pydantic models for the Agentic RAG search endpoint."""

from typing import Any, Literal

from pydantic import BaseModel, Field

WikiContextStatus = Literal[
    "disabled",
    "selected",
    "missing_pack",
    "no_match",
    "invalid_json",
    "invalid_schema",
]

WikiPromotionStatus = Literal[
    "disabled",
    "not_selected",
    "no_valid_source_refs",
    "promoted",
    "no_matching_source_refs",
    "fetch_failed",
]


class AgenticSearchQuery(BaseModel):
    """Request body for POST /collections/{id}/agentic-search."""

    question: str = Field(..., description="The question to answer")
    search_type: Literal["semantic", "keyword", "hybrid"] = Field(
        "hybrid", description="Search algorithm"
    )
    search_limit: int = Field(5, ge=1, le=50, description="Max documents per retrieval")
    min_score: float | None = Field(
        None, ge=0, le=1, description="Minimum relevance score threshold"
    )
    filter: dict[str, Any] | None = Field(None, description="Metadata filter")
    max_rewrites: int = Field(3, ge=0, le=10, description="Max query rewrite attempts")
    llm_provider: str | None = Field(
        None, description="LLM provider override: auto, openai, google, or ollama"
    )
    llm_model: str | None = Field(None, description="LLM model override")
    llm_temperature: float | None = Field(
        None, ge=0, le=2, description="LLM temperature override"
    )
    use_wiki_context: bool = Field(
        default=True,
        description="Use existing non-authoritative LLM Wiki navigation context during generation",
    )


class AgenticSearchResult(BaseModel):
    """Response body for the agentic search endpoint."""

    generation: str = Field("", description="Generated answer")
    relevant_documents: list[dict[str, Any]] = Field(
        default_factory=list, description="Documents deemed relevant"
    )
    steps: list[str] = Field(
        default_factory=list, description="Execution trace"
    )
    query_rewrites: list[str] = Field(
        default_factory=list, description="Query rewrite history"
    )
    rewrite_count: int = Field(0, description="Number of rewrites performed")
    error: str | None = Field(None, description="Error message if failed")
    no_context_found: bool = Field(
        default=False, description="True when no relevant context was found"
    )
    selected_wiki_pages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Selected non-authoritative wiki page metadata",
    )
    wiki_context_status: WikiContextStatus = Field(
        default="disabled",
        description="Finite status for optional wiki context resolution",
    )
    wiki_source_refs: list[dict[str, str]] = Field(
        default_factory=list,
        description="Structured wiki source refs selected for benchmark observability",
    )
    wiki_promotion_status: WikiPromotionStatus = Field(
        default="disabled",
        description="Finite status for optional wiki source-ref promotion",
    )
    wiki_promoted_document_ids: list[str] = Field(
        default_factory=list,
        description="Document IDs promoted from wiki source refs",
    )
    retrieved_document_ids: list[str] = Field(
        default_factory=list,
        description="Pre-grade retrieved document IDs after wiki promotion merge",
    )
