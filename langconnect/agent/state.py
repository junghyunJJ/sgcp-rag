"""Agentic RAG state definition.

Defines the TypedDict that flows through the LangGraph StateGraph.
Each node reads from and writes partial updates to this state.
"""

import operator
from typing import Annotated, Any, Literal, TypedDict


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
