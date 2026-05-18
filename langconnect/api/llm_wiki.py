"""REST endpoints for collection LLM Wiki rebuilds."""

from uuid import UUID

from fastapi import APIRouter

from langconnect.models.llm_wiki import LLMWikiRebuildRequest, LLMWikiRebuildResponse
from langconnect.services.llm_wiki import rebuild_llm_wiki

router = APIRouter(tags=["llm-wiki"])


@router.post(
    "/collections/{collection_id}/llm-wiki/rebuild",
)
async def llm_wiki_rebuild(
    collection_id: UUID,
    request: LLMWikiRebuildRequest | None = None,
) -> LLMWikiRebuildResponse:
    """Synchronously rebuild generated LLM Wiki artifacts for a collection."""
    request = request or LLMWikiRebuildRequest()
    return await rebuild_llm_wiki(
        str(collection_id),
        llm_provider=request.llm_provider,
        llm_model=request.llm_model,
        llm_temperature=request.llm_temperature,
    )
