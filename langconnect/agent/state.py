"""Agentic RAG state definition.

Defines the TypedDict that flows through the LangGraph StateGraph.
Each node reads from and writes partial updates to this state.
"""

from typing import Any, Literal, TypedDict


class AgentState(TypedDict):
    """State flowing through the Agentic RAG graph."""

    question: str
    collection_id: str
    user_id: str | None
    search_type: Literal["semantic", "keyword", "hybrid"]
    search_limit: int
    search_filter: dict[str, Any] | None
    documents: list[dict[str, Any]]
    relevant_documents: list[dict[str, Any]]
    generation: str
    query_rewrites: list[str]
    rewrite_count: int
    max_rewrites: int
    steps: list[str]
    error: str | None
