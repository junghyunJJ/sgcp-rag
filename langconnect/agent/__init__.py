"""Agentic RAG — self-correcting retrieval-augmented generation.

Main entry point: run_agentic_search()
Called by both the REST API and MCP tools.
"""

import asyncio
import logging
import os
from typing import Any, Literal

from langconnect.agent.config import get_agent_llm
from langconnect.agent.graph import build_agentic_rag_graph
from langconnect.agent.wiki_context import WikiContextResult, resolve_wiki_context

AGENTIC_SEARCH_TIMEOUT = 120  # seconds

logger = logging.getLogger(__name__)


async def run_agentic_search(  # noqa: PLR0913
    question: str,
    collection_id: str,
    *,
    search_type: Literal["semantic", "keyword", "hybrid"] = "hybrid",
    search_limit: int = 5,
    search_filter: dict[str, Any] | None = None,
    min_score: float | None = None,
    max_rewrites: int | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_temperature: float | None = None,
    use_wiki_context: bool = False,
) -> dict[str, Any]:
    """Run an agentic RAG search with self-correcting retrieval loop.

    Args:
        question: The user's question.
        collection_id: UUID of the collection to search.
        search_type: Search algorithm ("semantic", "keyword", "hybrid").
        search_limit: Max documents per retrieval.
        search_filter: Optional metadata filter dict.
        min_score: Optional minimum relevance score threshold.
        max_rewrites: Maximum query rewrite attempts (loop guard).
        llm_provider: LLM provider override ("openai" or "google").
        llm_model: LLM model name override.
        llm_temperature: LLM temperature override.
        use_wiki_context: Use non-authoritative LLM Wiki context during generation.

    Returns:
        Dict with keys: generation, relevant_documents, steps,
        query_rewrites, rewrite_count, error.
    """
    wiki_result = WikiContextResult(context="", selected_pages=[], status="disabled")

    try:
        if max_rewrites is None:
            max_rewrites = int(os.getenv("AGENT_MAX_REWRITES", "3"))

        if use_wiki_context:
            wiki_result = resolve_wiki_context(collection_id, question)

        llm = get_agent_llm(
            provider=llm_provider,
            model=llm_model,
            temperature=llm_temperature,
        )

        graph = build_agentic_rag_graph(llm)

        initial_state = {
            "question": question,
            "collection_id": collection_id,
            "search_type": search_type,
            "search_limit": search_limit,
            "search_filter": search_filter,
            "min_score": min_score,
            "documents": [],
            "relevant_documents": [],
            "generation": "",
            "query_rewrites": [],
            "rewrite_count": 0,
            "max_rewrites": max_rewrites,
            "steps": [],
            "error": None,
            "no_context_found": False,
            "use_wiki_context": use_wiki_context,
            "wiki_context": wiki_result.context,
            "selected_wiki_pages": wiki_result.selected_pages,
            "wiki_context_status": wiki_result.status,
        }

        result = await asyncio.wait_for(
            graph.ainvoke(initial_state),
            timeout=AGENTIC_SEARCH_TIMEOUT,
        )

        return {
            "generation": result.get("generation", ""),
            "relevant_documents": result.get("relevant_documents", []),
            "steps": result.get("steps", []),
            "query_rewrites": result.get("query_rewrites", []),
            "rewrite_count": result.get("rewrite_count", 0),
            "error": result.get("error"),
            "no_context_found": result.get("no_context_found", False),
            "selected_wiki_pages": result.get(
                "selected_wiki_pages",
                wiki_result.selected_pages,
            ),
            "wiki_context_status": result.get("wiki_context_status")
            or wiki_result.status,
        }

    except Exception as e:
        logger.exception("Agentic search failed")
        return {
            "generation": "",
            "relevant_documents": [],
            "steps": [f"error: {e!s}"],
            "query_rewrites": [],
            "rewrite_count": 0,
            "error": str(e),
            "no_context_found": False,
            "selected_wiki_pages": wiki_result.selected_pages,
            "wiki_context_status": wiki_result.status,
        }
