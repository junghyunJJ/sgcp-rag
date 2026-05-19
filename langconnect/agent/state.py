"""Agentic RAG state definition.

Defines the TypedDict that flows through the LangGraph StateGraph.
Each node reads from and writes partial updates to this state.
"""

import operator
from typing import Annotated, Any, Literal, TypedDict

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


class AgentState(TypedDict):
    """State flowing through the Agentic RAG graph."""

    question: str
    collection_id: str
    search_type: Literal["semantic", "keyword", "hybrid"]
    search_limit: int
    search_filter: dict[str, Any] | None
    min_score: float | None
    documents: list[dict[str, Any]]
    relevant_documents: list[dict[str, Any]]
    generation: str
    query_rewrites: Annotated[list[str], operator.add]
    rewrite_count: int
    max_rewrites: int
    steps: Annotated[list[str], operator.add]
    error: str | None
    no_context_found: bool
    use_wiki_context: bool
    wiki_context: str
    selected_wiki_pages: list[dict[str, Any]]
    wiki_context_status: WikiContextStatus
    wiki_source_refs: list[dict[str, str]]
    wiki_promoted_documents: list[dict[str, Any]]
    wiki_promotion_status: WikiPromotionStatus
