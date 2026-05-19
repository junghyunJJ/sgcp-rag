"""LLM Wiki context loading and selection for agentic RAG."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

WikiContextStatus = Literal[
    "disabled",
    "selected",
    "missing_pack",
    "no_match",
    "invalid_json",
    "invalid_schema",
]

DEFAULT_WIKI_CONTEXT_DIR = Path("llm_wiki/collections")
WIKI_CONTEXT_DIR_ENV = "LANGCONNECT_WIKI_CONTEXT_DIR"
MAX_SELECTED_PAGES = 3
MAX_WIKI_SOURCE_REFS = 8
LOW_SIGNAL_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "cell",
    "cells",
    "does",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "type",
    "types",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
    "without",
}


@dataclass(frozen=True)
class WikiContextResult:
    """Resolved wiki context and public metadata for one request."""

    context: str
    selected_pages: list[dict[str, Any]]
    status: WikiContextStatus


def _empty(status: WikiContextStatus) -> WikiContextResult:
    return WikiContextResult(context="", selected_pages=[], status=status)


def _wiki_dir(path: Path | str | None) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.getenv(WIKI_CONTEXT_DIR_ENV, str(DEFAULT_WIKI_CONTEXT_DIR)))


def _pack_path(
    collection_id: str,
    wiki_dir: Path | str | None,
) -> Path | None:
    base_dir = _wiki_dir(wiki_dir).resolve()
    if (
        not collection_id
        or "/" in collection_id
        or "\\" in collection_id
        or Path(collection_id).name != collection_id
    ):
        return None

    path = (base_dir / f"{collection_id}.json").resolve()
    if not path.is_relative_to(base_dir):
        return None
    return path


def _tokenize(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", text.casefold()))
    return tokens - LOW_SIGNAL_TOKENS


def _require_text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _validate_source_refs(value: object) -> list[dict[str, str]] | None:
    if not isinstance(value, list) or not value:
        return None

    refs: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        file_id = _require_text(item.get("file_id"))
        chunk_id = _require_text(item.get("chunk_id"))
        if file_id is None or chunk_id is None:
            return None
        refs.append({"file_id": file_id, "chunk_id": chunk_id})
    return refs


def _validate_page(page: object) -> dict[str, Any] | None:
    if not isinstance(page, dict):
        return None

    page_id = _require_text(page.get("id"))
    title = _require_text(page.get("title"))
    summary = _require_text(page.get("summary"))
    keywords = page.get("keywords")
    source_refs = _validate_source_refs(page.get("source_refs"))
    if (
        page_id is None
        or title is None
        or summary is None
        or not isinstance(keywords, list)
        or source_refs is None
    ):
        return None
    if not all(isinstance(keyword, str) for keyword in keywords):
        return None

    return {
        "id": page_id,
        "title": title,
        "summary": summary,
        "keywords": [keyword.strip() for keyword in keywords if keyword.strip()],
        "source_refs": source_refs,
    }


def _validate_pages(pack: object, collection_id: str) -> list[dict[str, Any]] | None:
    if not isinstance(pack, dict) or pack.get("collection_id") != collection_id:
        return None

    pages = pack.get("pages")
    if not isinstance(pages, list):
        return None

    validated_pages: list[dict[str, Any]] = []
    for page in pages:
        validated_page = _validate_page(page)
        if validated_page is None:
            return None
        validated_pages.append(validated_page)

    return validated_pages


def _page_tokens(page: dict[str, Any]) -> set[str]:
    keywords = " ".join(str(keyword) for keyword in page["keywords"])
    return _tokenize(f"{page['title']} {page['summary']} {keywords}")


def _select_pages(
    pages: list[dict[str, Any]],
    question: str,
    *,
    limit: int = MAX_SELECTED_PAGES,
) -> list[dict[str, Any]]:
    question_tokens = _tokenize(question)
    scored_pages: list[tuple[int, str, str, dict[str, Any]]] = []
    for page in pages:
        score = len(question_tokens & _page_tokens(page))
        if score > 0:
            scored_pages.append(
                (score, page["title"].casefold(), page["id"].casefold(), page)
            )

    scored_pages.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected_pages: list[dict[str, Any]] = []
    for score, _, _, page in scored_pages[:limit]:
        selected_pages.append(
            {
                "id": page["id"],
                "title": page["title"],
                "source_refs": page["source_refs"],
                "score": score,
                "summary": page["summary"],
            }
        )
    return selected_pages


def _render_context(selected_pages: list[dict[str, Any]]) -> str:
    lines = [
        "LLM Wiki context (non-authoritative navigation memory; not evidence):",
    ]
    for index, page in enumerate(selected_pages, start=1):
        refs = ", ".join(
            f"{ref['file_id']}:{ref['chunk_id']}" for ref in page.get("source_refs", [])
        )
        lines.extend(
            [
                f"{index}. {page['title']}",
                f"   Summary: {page['summary']}",
                f"   Navigation source refs: {refs}",
            ]
        )
    return "\n".join(lines)


def extract_wiki_source_refs(
    selected_pages: list[dict[str, Any]],
    *,
    limit: int = MAX_WIKI_SOURCE_REFS,
) -> list[dict[str, str]]:
    """Flatten selected wiki page source refs for bounded chunk promotion."""
    if limit <= 0:
        return []

    refs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for page in selected_pages:
        page_refs = page.get("source_refs", []) if isinstance(page, dict) else []
        if not isinstance(page_refs, list):
            continue
        for item in page_refs:
            if not isinstance(item, dict):
                continue
            file_id = _require_text(item.get("file_id"))
            chunk_id = _require_text(item.get("chunk_id"))
            if file_id is None or chunk_id is None:
                continue
            key = (file_id, chunk_id)
            if key in seen:
                continue
            seen.add(key)
            refs.append({"file_id": file_id, "chunk_id": chunk_id})
            if len(refs) >= limit:
                return refs
    return refs


def _load_pack(path: Path) -> tuple[object | None, WikiContextStatus | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, "invalid_json"
    except OSError:
        return None, "invalid_schema"


def resolve_wiki_context(
    collection_id: str,
    question: str,
    *,
    wiki_dir: Path | str | None = None,
) -> WikiContextResult:
    """Load and render selected wiki context for a collection/question pair."""
    path = _pack_path(collection_id, wiki_dir)
    if path is None:
        return _empty("invalid_schema")
    if not path.exists():
        return _empty("missing_pack")

    pack, load_error = _load_pack(path)
    if load_error is not None:
        return _empty(load_error)

    pages = _validate_pages(pack, collection_id)
    if pages is None:
        return _empty("invalid_schema")

    selected_pages = _select_pages(pages, question)
    if not selected_pages:
        return _empty("no_match")

    return WikiContextResult(
        context=_render_context(selected_pages),
        selected_pages=selected_pages,
        status="selected",
    )
