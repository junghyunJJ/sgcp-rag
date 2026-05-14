"""Agentic RAG search endpoint.

POST /collections/{collection_id}/agentic-search

This endpoint runs a self-correcting retrieval loop that automatically
evaluates retrieved documents, rewrites queries when needed, and validates
generated answers. The existing /documents/search endpoint is unchanged.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException

from langconnect.agent import run_agentic_search
from langconnect.models.agentic import AgenticSearchQuery, AgenticSearchResult

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agentic"])


@router.post(
    "/collections/{collection_id}/agentic-search",
    response_model=AgenticSearchResult,
)
async def agentic_search(
    collection_id: UUID,
    query: AgenticSearchQuery,
):
    """Run an agentic RAG search with self-correcting retrieval."""
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = await run_agentic_search(
        question=query.question,
        collection_id=str(collection_id),
        search_type=query.search_type,
        search_limit=query.search_limit,
        search_filter=query.filter,
        min_score=query.min_score,
        max_rewrites=query.max_rewrites,
        llm_provider=query.llm_provider,
        llm_model=query.llm_model,
        llm_temperature=query.llm_temperature,
        use_wiki_context=query.use_wiki_context,
    )

    return AgenticSearchResult(**result)
