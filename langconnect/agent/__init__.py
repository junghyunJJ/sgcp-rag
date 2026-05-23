"""Agentic RAG — self-correcting retrieval-augmented generation.

Main entry point: run_agentic_search()
Called by both the REST API and MCP tools.
"""

import asyncio
import logging
import os
from typing import Any, Literal

import httpx

from langconnect.agent.config import (
    DEFAULT_AGENT_OLLAMA_MODEL,
    DEFAULT_AGENT_OPENAI_MODEL,
    get_agent_llm,
    get_agent_ollama_base_url,
    is_ollama_model_available,
)
from langconnect.agent.graph import build_agentic_rag_graph
from langconnect.agent.wiki_context import (
    WikiContextResult,
    extract_wiki_source_refs,
    resolve_wiki_context,
)
from langconnect.database.collections import Collection

AGENTIC_SEARCH_TIMEOUT = 120  # seconds
DEFAULT_AGENT_PROVIDER = "openai"
AGENT_PROVIDER_AUTO = "auto"

logger = logging.getLogger(__name__)


def _agent_llm_provider(provider: str | None = None) -> str:
    return (
        (provider or os.getenv("AGENT_LLM_PROVIDER", DEFAULT_AGENT_PROVIDER))
        .strip()
        .lower()
    )


def _agent_ollama_model(model: str | None = None) -> str:
    return model or os.getenv("AGENT_LLM_MODEL", DEFAULT_AGENT_OLLAMA_MODEL)


def _agent_openai_fallback_model() -> str:
    return os.getenv("AGENT_LLM_OPENAI_MODEL", DEFAULT_AGENT_OPENAI_MODEL)


def _format_agentic_result(
    result: dict[str, Any],
    wiki_result: WikiContextResult,
) -> dict[str, Any]:
    documents = result.get("documents", [])
    promoted_documents = result.get("wiki_promoted_documents", [])
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
        "wiki_context_status": result.get("wiki_context_status") or wiki_result.status,
        "wiki_source_refs": result.get("wiki_source_refs", []),
        "wiki_promotion_status": result.get("wiki_promotion_status", "disabled"),
        "wiki_promoted_document_ids": _document_ids(promoted_documents),
        "retrieved_document_ids": _document_ids(documents),
    }


def _format_agentic_error(
    error: Exception,
    wiki_result: WikiContextResult,
) -> dict[str, Any]:
    return {
        "generation": "",
        "relevant_documents": [],
        "steps": [f"error: {error!s}"],
        "query_rewrites": [],
        "rewrite_count": 0,
        "error": str(error),
        "no_context_found": False,
        "selected_wiki_pages": wiki_result.selected_pages,
        "wiki_context_status": wiki_result.status,
        "wiki_source_refs": [],
        "wiki_promotion_status": "disabled",
        "wiki_promoted_document_ids": [],
        "retrieved_document_ids": [],
    }


def _document_ids(documents: object) -> list[str]:
    if not isinstance(documents, list):
        return []

    ids: list[str] = []
    for document in documents:
        if not isinstance(document, dict):
            continue
        document_id = document.get("id")
        if document_id is None:
            continue
        ids.append(str(document_id))
    return ids


async def _resolve_wiki_promotion(
    collection_id: str,
    wiki_result: WikiContextResult,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], str, list[str]]:
    if wiki_result.status != "selected":
        return [], [], "not_selected", []

    source_refs = extract_wiki_source_refs(wiki_result.selected_pages)
    if not source_refs:
        return [], [], "no_valid_source_refs", []

    try:
        promoted_documents = await Collection(
            collection_id,
        ).get_many_by_source_refs(source_refs)
    except Exception as error:
        logger.warning(
            "LLM Wiki source-ref promotion failed for collection %s: %s",
            collection_id,
            error,
            exc_info=True,
        )
        return source_refs, [], "fetch_failed", ["wiki_promotion: fetch_failed"]

    status = "promoted" if promoted_documents else "no_matching_source_refs"
    return source_refs, promoted_documents, status, []


async def _invoke_agent_graph(
    *,
    provider: str,
    model: str | None,
    temperature: float | None,
    initial_state: dict[str, Any],
    base_url: str | None = None,
) -> dict[str, Any]:
    llm = get_agent_llm(
        provider=provider,
        model=model,
        temperature=temperature,
        base_url=base_url,
    )
    graph = build_agentic_rag_graph(llm)
    return await asyncio.wait_for(
        graph.ainvoke(initial_state),
        timeout=AGENTIC_SEARCH_TIMEOUT,
    )


def _url_is_under_base(url: object, base_url: str) -> bool:
    current = str(url).rstrip("/")
    expected = base_url.rstrip("/")
    return current == expected or current.startswith(f"{expected}/")


def _safe_getattr(value: object, name: str) -> object | None:
    try:
        return getattr(value, name, None)
    except RuntimeError:
        return None


def _httpx_error_targets_base(error: BaseException, base_url: str) -> bool:
    for source in (_safe_getattr(error, "request"), _safe_getattr(error, "response")):
        request = _safe_getattr(source, "request") or source
        url = _safe_getattr(request, "url")
        if url is not None and _url_is_under_base(url, base_url):
            return True
    return False


def _is_agent_ollama_fallback_eligible(
    error: Exception,
    base_url: str,
) -> bool:
    current: BaseException | None = error
    while current is not None:
        if type(current).__module__.startswith("ollama"):
            return True

        if isinstance(current, httpx.HTTPError) and _httpx_error_targets_base(
            current,
            base_url,
        ):
            return True

        message = str(current).lower()
        if "ollama" in message and any(
            marker in message
            for marker in ("connect", "failed", "refused", "timeout", "unavailable")
        ):
            return True

        current = current.__cause__ or current.__context__

    return False


async def _run_agentic_search_auto(
    *,
    llm_model: str | None,
    llm_temperature: float | None,
    initial_state: dict[str, Any],
) -> dict[str, Any]:
    ollama_model = _agent_ollama_model(llm_model)
    ollama_base_url = get_agent_ollama_base_url()

    if await is_ollama_model_available(ollama_model, ollama_base_url):
        try:
            return await _invoke_agent_graph(
                provider="ollama",
                model=ollama_model,
                temperature=llm_temperature,
                base_url=ollama_base_url,
                initial_state=initial_state,
            )
        except Exception as error:
            if not _is_agent_ollama_fallback_eligible(error, ollama_base_url):
                raise
            logger.warning(
                "Ollama Agentic RAG failed; falling back to OpenAI: %s",
                error,
            )

    return await _invoke_agent_graph(
        provider="openai",
        model=_agent_openai_fallback_model(),
        temperature=llm_temperature,
        initial_state=initial_state,
    )


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
    use_wiki_context: bool = True,
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
        llm_provider: LLM provider override ("auto", "openai", "google", or "ollama").
        llm_model: LLM model name override.
        llm_temperature: LLM temperature override.
        use_wiki_context: Use existing non-authoritative LLM Wiki context during generation.

    Returns:
        Dict with keys: generation, relevant_documents, steps,
        query_rewrites, rewrite_count, error.
    """
    wiki_result = WikiContextResult(context="", selected_pages=[], status="disabled")
    wiki_source_refs: list[dict[str, str]] = []
    wiki_promoted_documents: list[dict[str, Any]] = []
    wiki_promotion_status = "disabled"
    wiki_promotion_steps: list[str] = []

    try:
        if max_rewrites is None:
            max_rewrites = int(os.getenv("AGENT_MAX_REWRITES", "3"))

        if use_wiki_context:
            wiki_result = resolve_wiki_context(collection_id, question)
            (
                wiki_source_refs,
                wiki_promoted_documents,
                wiki_promotion_status,
                wiki_promotion_steps,
            ) = await _resolve_wiki_promotion(collection_id, wiki_result)

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
            "steps": wiki_promotion_steps,
            "error": None,
            "no_context_found": False,
            "use_wiki_context": use_wiki_context,
            "wiki_context": "",
            "selected_wiki_pages": wiki_result.selected_pages,
            "wiki_context_status": wiki_result.status,
            "wiki_source_refs": wiki_source_refs,
            "wiki_promoted_documents": wiki_promoted_documents,
            "wiki_promotion_status": wiki_promotion_status,
        }

        provider = _agent_llm_provider(llm_provider)
        if provider == AGENT_PROVIDER_AUTO:
            result = await _run_agentic_search_auto(
                llm_model=llm_model,
                llm_temperature=llm_temperature,
                initial_state=initial_state,
            )
        else:
            result = await _invoke_agent_graph(
                provider=provider,
                model=llm_model,
                temperature=llm_temperature,
                base_url=get_agent_ollama_base_url() if provider == "ollama" else None,
                initial_state=initial_state,
            )

        return _format_agentic_result(result, wiki_result)

    except Exception as e:
        logger.exception("Agentic search failed")
        return _format_agentic_error(e, wiki_result)
