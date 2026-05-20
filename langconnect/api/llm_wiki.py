"""REST endpoints for collection LLM Wiki artifacts."""

from uuid import UUID

from fastapi import APIRouter, HTTPException

from langconnect.models.llm_wiki import (
    LLMWikiIndexResponse,
    LLMWikiPageResponse,
    LLMWikiRebuildRequest,
    LLMWikiRebuildResponse,
)
from langconnect.services.llm_wiki import (
    LLMWikiArtifactError,
    read_llm_wiki_index,
    read_llm_wiki_page,
    rebuild_llm_wiki,
)

router = APIRouter(tags=["llm-wiki"])


def _raise_artifact_http_error(error: LLMWikiArtifactError) -> None:
    raise HTTPException(
        status_code=error.status_code,
        detail={"code": error.code, "message": error.message},
    )


@router.get(
    "/collections/{collection_id}/llm-wiki",
)
async def llm_wiki_index(collection_id: UUID) -> LLMWikiIndexResponse:
    """Read generated LLM Wiki index and navigation metadata for a collection."""
    try:
        return read_llm_wiki_index(str(collection_id))
    except LLMWikiArtifactError as error:
        _raise_artifact_http_error(error)


@router.get(
    "/collections/{collection_id}/llm-wiki/pages/{section}/{slug}",
)
async def llm_wiki_page(
    collection_id: UUID,
    section: str,
    slug: str,
) -> LLMWikiPageResponse:
    """Read one generated source or concept page for a collection."""
    try:
        return read_llm_wiki_page(str(collection_id), section, slug)
    except LLMWikiArtifactError as error:
        _raise_artifact_http_error(error)


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
