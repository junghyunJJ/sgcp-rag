"""Synchronous collection-level LLM Wiki rebuild service."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from langconnect.agent.config import get_agent_llm
from langconnect.agent.wiki_context import _validate_pages
from langconnect.database.collections import Collection
from langconnect.models.llm_wiki import LLMWikiRebuildResponse

COLLECTION_PAGE_SIZE = 100
SOURCE_MAX_CHUNKS = 12
SOURCE_CHUNK_CHAR_LIMIT = 2000
SOURCE_INPUT_CHAR_LIMIT = 24000
SOURCE_SUMMARY_CHAR_LIMIT = 1200
SOURCE_REF_LIMIT = 5

CONCEPT_MAX_PAGES = 10
CONCEPT_INPUT_SOURCE_LIMIT = 100
CONCEPT_INPUT_CHAR_LIMIT = 50000
CONCEPT_SUMMARY_CHAR_LIMIT = 1200
CONCEPT_KEYWORD_LIMIT = 12

DEFAULT_LLM_WIKI_ROOT = Path("llm_wiki")
CONFIDENCE_VALUES = {"low", "medium", "high"}


class LLMWikiRebuildError(RuntimeError):
    """Raised when a collection LLM Wiki rebuild cannot be published."""


@dataclass(frozen=True)
class _Chunk:
    id: str
    content: str
    metadata: dict[str, Any]
    file_id: str
    source: str


@dataclass(frozen=True)
class _Page:
    id: str
    title: str
    page_type: Literal["source", "concept"]
    summary: str
    keywords: list[str]
    source_refs: list[dict[str, str]]
    confidence: Literal["low", "medium", "high"]
    relative_path: str
    reference_count: int
    chunk_count: int = 0
    file_id: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class _ArtifactBundle:
    collection_id: str
    generated_at: str
    chunk_count: int
    source_pages: list[_Page]
    concept_pages: list[_Page]
    public_pack_path: Path


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _slugify(value: object, *, fallback: str) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:80] or fallback


def _unique_slug(slug: str, used: set[str]) -> str:
    candidate = slug
    index = 2
    while candidate in used:
        candidate = f"{slug}-{index}"
        index += 1
    used.add(candidate)
    return candidate


def _safe_join(base_dir: Path, relative_path: str) -> Path:
    base = base_dir.resolve()
    path = (base / relative_path).resolve()
    if not path.is_relative_to(base):
        raise LLMWikiRebuildError(f"Unsafe generated path: {relative_path}")
    return path


def _clip_text(value: object, limit: int) -> str:
    text = str(value or "").strip()
    return text[:limit]


def _coerce_keywords(value: object, *, limit: int = CONCEPT_KEYWORD_LIMIT) -> list[str]:
    if not isinstance(value, list):
        return []
    keywords: list[str] = []
    for item in value:
        keyword = str(item).strip()
        if keyword and keyword not in keywords:
            keywords.append(keyword[:80])
        if len(keywords) >= limit:
            break
    return keywords


def _coerce_confidence(value: object) -> Literal["low", "medium", "high"]:
    confidence = str(value or "medium").strip().lower()
    if confidence in CONFIDENCE_VALUES:
        return confidence  # type: ignore[return-value]
    return "medium"


def _response_content(response: object) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    return json.dumps(content)


def _parse_json_response(response: object) -> object:
    content = _response_content(response).strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError as error:
        raise LLMWikiRebuildError(f"LLM returned invalid JSON: {error}") from error


def _as_chunk(raw: dict[str, Any], fallback_index: int) -> _Chunk:
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    chunk_id = str(raw.get("id") or raw.get("uuid") or f"chunk-{fallback_index}")
    content = str(raw.get("content") or raw.get("page_content") or "")
    source = str(
        metadata.get("source")
        or metadata.get("filename")
        or metadata.get("title")
        or "unknown-source",
    )
    file_id = str(metadata.get("file_id") or source or f"source-{fallback_index}")
    return _Chunk(
        id=chunk_id,
        content=content,
        metadata=dict(metadata),
        file_id=file_id,
        source=source,
    )


async def _list_all_chunks(collection_id: str) -> list[_Chunk]:
    collection = Collection(collection_id=collection_id)
    chunks: list[_Chunk] = []
    offset = 0
    while True:
        page = await collection.list(limit=COLLECTION_PAGE_SIZE, offset=offset)
        chunks.extend(_as_chunk(raw, len(chunks)) for raw in page)
        if len(page) < COLLECTION_PAGE_SIZE:
            break
        offset += COLLECTION_PAGE_SIZE
    return chunks


def _group_chunks_by_source(chunks: list[_Chunk]) -> list[tuple[str, list[_Chunk]]]:
    grouped: dict[str, list[_Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.file_id, []).append(chunk)
    return [
        (file_id, sorted(items, key=lambda item: item.id))
        for file_id, items in sorted(grouped.items(), key=lambda item: item[0])
    ]


def _bounded_chunk_blocks(chunks: list[_Chunk]) -> tuple[str, list[_Chunk]]:
    selected = chunks[:SOURCE_MAX_CHUNKS]
    blocks: list[str] = []
    budget = SOURCE_INPUT_CHAR_LIMIT
    per_chunk_limit = min(
        SOURCE_CHUNK_CHAR_LIMIT,
        max(1, (SOURCE_INPUT_CHAR_LIMIT // SOURCE_MAX_CHUNKS) - 64),
    )
    for chunk in selected:
        content = _clip_text(chunk.content, per_chunk_limit)
        block = f'<chunk id="{chunk.id}">\n{content}\n</chunk>'
        if blocks and len("\n\n".join([*blocks, block])) > budget:
            break
        blocks.append(block)
    return "\n\n".join(blocks), selected[: len(blocks)]


def _source_prompt(file_id: str, chunks: list[_Chunk]) -> tuple[str, list[_Chunk]]:
    chunk_blocks, selected_chunks = _bounded_chunk_blocks(chunks)
    source = chunks[0].source if chunks else file_id
    prompt = f"""Build one source page for an LLM Wiki.

Return one JSON object with keys: title, summary, keywords, confidence.
The summary must be concise and non-authoritative navigation memory.
Confidence must be low, medium, or high.

Source file_id: {file_id}
Source label: {source}

Chunks:
{chunk_blocks}
"""
    return prompt[:SOURCE_INPUT_CHAR_LIMIT], selected_chunks


def _concept_prompt(source_pages: list[_Page]) -> str:
    entries: list[str] = []
    total = 0
    for page in source_pages[:CONCEPT_INPUT_SOURCE_LIMIT]:
        refs = ", ".join(
            f"{ref['file_id']}:{ref['chunk_id']}" for ref in page.source_refs
        )
        entry = (
            f"- id: {page.id}\n"
            f"  title: {page.title}\n"
            f"  summary: {_clip_text(page.summary, SOURCE_SUMMARY_CHAR_LIMIT)}\n"
            f"  keywords: {', '.join(page.keywords)}\n"
            f"  refs: {refs}\n"
        )
        if entries and total + len(entry) > CONCEPT_INPUT_CHAR_LIMIT:
            break
        entries.append(entry)
        total += len(entry)

    return f"""Synthesize up to {CONCEPT_MAX_PAGES} concept pages for an LLM Wiki.

Return JSON as {{"concepts": [{{"title": "...", "summary": "...", "keywords": ["..."], "source_refs": [{{"file_id": "...", "chunk_id": "..."}}], "confidence": "medium"}}]}}.
Concept pages are non-authoritative navigation memory only.

Source pages:
{''.join(entries)}
"""[:CONCEPT_INPUT_CHAR_LIMIT]


async def _invoke_json(llm: object, prompt: str) -> object:
    if not hasattr(llm, "ainvoke"):
        raise LLMWikiRebuildError("Configured LLM does not support async invocation")
    response = await llm.ainvoke(prompt)
    return _parse_json_response(response)


def _source_refs(chunks: list[_Chunk]) -> list[dict[str, str]]:
    return [
        {"file_id": chunk.file_id, "chunk_id": chunk.id}
        for chunk in chunks[:SOURCE_REF_LIMIT]
    ]


def _page_pack_record(page: _Page) -> dict[str, Any]:
    return {
        "id": page.id,
        "type": page.page_type,
        "title": page.title,
        "summary": page.summary,
        "keywords": page.keywords,
        "source_refs": page.source_refs,
        "path": page.relative_path,
    }


async def _generate_source_pages(
    llm: object,
    groups: list[tuple[str, list[_Chunk]]],
) -> list[_Page]:
    pages: list[_Page] = []
    used_slugs: set[str] = set()
    for file_id, chunks in groups:
        prompt, selected_chunks = _source_prompt(file_id, chunks)
        data = await _invoke_json(llm, prompt)
        if not isinstance(data, dict):
            raise LLMWikiRebuildError("LLM source response must be a JSON object")
        title = _clip_text(data.get("title") or chunks[0].source or file_id, 160)
        summary = _clip_text(data.get("summary"), SOURCE_SUMMARY_CHAR_LIMIT)
        if not title or not summary:
            raise LLMWikiRebuildError("LLM source response missing title or summary")
        slug = _unique_slug(_slugify(title, fallback="source"), used_slugs)
        refs = _source_refs(selected_chunks)
        if not refs:
            raise LLMWikiRebuildError(f"Source {file_id} has no chunk references")
        pages.append(
            _Page(
                id=f"source-{slug}",
                title=title,
                page_type="source",
                summary=summary,
                keywords=_coerce_keywords(data.get("keywords")),
                source_refs=refs,
                confidence=_coerce_confidence(data.get("confidence")),
                relative_path=f"sources/{slug}.md",
                reference_count=len(refs),
                chunk_count=len(chunks),
                file_id=file_id,
                source=chunks[0].source if chunks else file_id,
            )
        )
    return pages


def _valid_concept_refs(
    refs: object,
    allowed_refs: set[tuple[str, str]],
) -> list[dict[str, str]]:
    valid_refs: list[dict[str, str]] = []
    if isinstance(refs, list):
        for item in refs:
            if not isinstance(item, dict):
                continue
            file_id = str(item.get("file_id") or "").strip()
            chunk_id = str(item.get("chunk_id") or "").strip()
            if (file_id, chunk_id) in allowed_refs:
                valid_refs.append({"file_id": file_id, "chunk_id": chunk_id})
            if len(valid_refs) >= SOURCE_REF_LIMIT:
                break
    return valid_refs


async def _generate_concept_pages(
    llm: object,
    source_pages: list[_Page],
) -> list[_Page]:
    if not source_pages:
        return []
    data = await _invoke_json(llm, _concept_prompt(source_pages))
    concepts = data.get("concepts") if isinstance(data, dict) else data
    if not isinstance(concepts, list):
        raise LLMWikiRebuildError("LLM concept response must contain a concepts list")

    allowed_refs = {
        (ref["file_id"], ref["chunk_id"])
        for page in source_pages
        for ref in page.source_refs
    }
    pages: list[_Page] = []
    used_slugs: set[str] = set()
    for concept in concepts[:CONCEPT_MAX_PAGES]:
        if not isinstance(concept, dict):
            raise LLMWikiRebuildError("LLM concept item must be an object")
        title = _clip_text(concept.get("title"), 160)
        summary = _clip_text(concept.get("summary"), CONCEPT_SUMMARY_CHAR_LIMIT)
        if not title or not summary:
            raise LLMWikiRebuildError("LLM concept response missing title or summary")
        slug = _unique_slug(_slugify(title, fallback="concept"), used_slugs)
        refs = _valid_concept_refs(concept.get("source_refs"), allowed_refs)
        if not refs:
            raise LLMWikiRebuildError("LLM concept response has no valid source_refs")
        pages.append(
            _Page(
                id=f"concept-{slug}",
                title=title,
                page_type="concept",
                summary=summary,
                keywords=_coerce_keywords(concept.get("keywords")),
                source_refs=refs,
                confidence=_coerce_confidence(concept.get("confidence")),
                relative_path=f"concepts/{slug}.md",
                reference_count=len(refs),
            )
        )
    return pages


def _yaml_string(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _frontmatter(page: _Page, generated_at: str) -> str:
    lines = [
        "---",
        f"title: {_yaml_string(page.title)}",
        f"type: {_yaml_string(page.page_type)}",
        f"summary: {_yaml_string(page.summary)}",
        "keywords:",
    ]
    lines.extend(f"  - {_yaml_string(keyword)}" for keyword in page.keywords)
    lines.append("source_refs:")
    for ref in page.source_refs:
        lines.extend(
            [
                f"  - file_id: {_yaml_string(ref['file_id'])}",
                f"    chunk_id: {_yaml_string(ref['chunk_id'])}",
            ]
        )
    lines.extend(
        [
            f"generated_at: {_yaml_string(generated_at)}",
            f"updated_at: {_yaml_string(generated_at)}",
            f"confidence: {_yaml_string(page.confidence)}",
            "---",
        ]
    )
    return "\n".join(lines)


def _render_page(page: _Page, generated_at: str) -> str:
    refs = "\n".join(
        f"- `{ref['file_id']}:{ref['chunk_id']}`" for ref in page.source_refs
    )
    keywords = ", ".join(page.keywords) or "none"
    return f"""{_frontmatter(page, generated_at)}

# {page.title}

> Generated LLM Wiki navigation memory. This page is replaceable on full rebuild and is not authoritative evidence.

## Summary
{page.summary}

## Keywords
{keywords}

## Navigation Source References
{refs}
"""


def _schema_markdown(collection_id: str, generated_at: str) -> str:
    return f"""# LLM Wiki Schema

Generated for collection `{collection_id}` at `{generated_at}`.

Generated source and concept pages are replaceable on full rebuild. They are non-authoritative navigation memory only; raw retrieved chunks remain the evidence source.

Required frontmatter fields for generated source/concept pages:

- `title`: string
- `type`: `source` or `concept`
- `summary`: string
- `keywords`: list of strings
- `source_refs`: list of `file_id` and `chunk_id` pairs
- `generated_at`: ISO timestamp string
- `updated_at`: ISO timestamp string
- `confidence`: `low`, `medium`, or `high`

Runtime pack shape:

```json
{{"collection_id": "...", "pages": [{{"id": "...", "title": "...", "summary": "...", "keywords": [], "source_refs": []}}]}}
```
"""


def _index_markdown(
    collection_id: str,
    generated_at: str,
    source_pages: list[_Page],
    concept_pages: list[_Page],
) -> str:
    lines = [
        f"# LLM Wiki Index: {collection_id}",
        "",
        f"Generated at `{generated_at}`.",
        "",
        "Generated files are replaceable. Use raw retrieved chunks as evidence.",
        "",
        "## Sources",
        "",
    ]
    if not source_pages:
        lines.append("_No sources indexed._")
    for page in source_pages:
        keywords = ", ".join(page.keywords) or "none"
        lines.append(
            f"- [{page.title}]({page.relative_path}) - {page.summary} "
            f"(keywords: {keywords}; chunks: {page.chunk_count})"
        )

    lines.extend(["", "## Concepts", ""])
    if not concept_pages:
        lines.append("_No concepts generated._")
    for page in concept_pages:
        keywords = ", ".join(page.keywords) or "none"
        lines.append(
            f"- [{page.title}]({page.relative_path}) - {page.summary} "
            f"(keywords: {keywords}; source refs: {page.reference_count})"
        )
    return "\n".join(lines) + "\n"


def _log_markdown(bundle: _ArtifactBundle) -> str:
    return f"""# Latest LLM Wiki Rebuild

- status: successful
- generated_at: {bundle.generated_at}
- collection_id: {bundle.collection_id}
- source_count: {len(bundle.source_pages)}
- chunk_count: {bundle.chunk_count}
- concept_count: {len(bundle.concept_pages)}
- pack_path: {bundle.public_pack_path}
"""


def _manifest(bundle: _ArtifactBundle) -> dict[str, Any]:
    pages = [*bundle.source_pages, *bundle.concept_pages]
    return {
        "collection_id": bundle.collection_id,
        "generated_at": bundle.generated_at,
        "source_count": len(bundle.source_pages),
        "chunk_count": bundle.chunk_count,
        "page_count": len(pages),
        "concept_count": len(bundle.concept_pages),
        "runtime_pack_path": str(bundle.public_pack_path),
        "sources": [
            {
                "type": page.page_type,
                "title": page.title,
                "path": page.relative_path,
                "slug": Path(page.relative_path).stem,
                "id": page.id,
                "file_id": page.file_id,
                "source": page.source,
                "chunk_count": page.chunk_count,
                "reference_count": page.reference_count,
            }
            for page in bundle.source_pages
        ],
        "concepts": [
            {
                "type": page.page_type,
                "title": page.title,
                "path": page.relative_path,
                "slug": Path(page.relative_path).stem,
                "id": page.id,
                "reference_count": page.reference_count,
            }
            for page in bundle.concept_pages
        ],
        "pages": [_page_pack_record(page) for page in pages],
    }


def _runtime_pack(
    collection_id: str,
    generated_at: str,
    pages: list[_Page],
) -> dict[str, Any]:
    return {
        "collection_id": collection_id,
        "generated_at": generated_at,
        "pages": [_page_pack_record(page) for page in pages],
    }


def _write_artifacts(
    staging_collection_dir: Path,
    staging_pack_path: Path,
    bundle: _ArtifactBundle,
) -> None:
    pages = [*bundle.source_pages, *bundle.concept_pages]
    for subdir in ("sources", "concepts"):
        (staging_collection_dir / subdir).mkdir(parents=True, exist_ok=True)
    for page in pages:
        path = _safe_join(staging_collection_dir, page.relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_render_page(page, bundle.generated_at), encoding="utf-8")

    (staging_collection_dir / "SCHEMA.md").write_text(
        _schema_markdown(bundle.collection_id, bundle.generated_at),
        encoding="utf-8",
    )
    (staging_collection_dir / "index.md").write_text(
        _index_markdown(
            bundle.collection_id,
            bundle.generated_at,
            bundle.source_pages,
            bundle.concept_pages,
        ),
        encoding="utf-8",
    )
    (staging_collection_dir / "log.md").write_text(
        _log_markdown(bundle),
        encoding="utf-8",
    )
    (staging_collection_dir / "manifest.json").write_text(
        json.dumps(
            _manifest(bundle),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    staging_pack_path.write_text(
        json.dumps(
            _runtime_pack(bundle.collection_id, bundle.generated_at, pages),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _validate_runtime_pack(pack_path: Path, collection_id: str) -> bool:
    try:
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return _validate_pages(pack, collection_id) is not None


def _restore_previous(
    *,
    public_collection_dir: Path,
    backup_collection_dir: Path | None,
    public_pack_path: Path,
    backup_pack_path: Path | None,
) -> None:
    if public_collection_dir.exists():
        shutil.rmtree(public_collection_dir)
    if backup_collection_dir and backup_collection_dir.exists():
        shutil.move(str(backup_collection_dir), str(public_collection_dir))
    if backup_pack_path and backup_pack_path.exists():
        shutil.copy2(backup_pack_path, public_pack_path)


def _publish_artifacts(
    *,
    staging_collection_dir: Path,
    staging_pack_path: Path,
    public_collection_dir: Path,
    public_pack_path: Path,
) -> None:
    token = uuid4().hex
    backup_collection_dir: Path | None = None
    backup_pack_path: Path | None = None

    try:
        if public_collection_dir.exists():
            backup_collection_dir = public_collection_dir.with_name(
                f".{public_collection_dir.name}.backup-{token}"
            )
            shutil.move(str(public_collection_dir), str(backup_collection_dir))

        if public_pack_path.exists():
            backup_pack_path = public_pack_path.with_name(
                f".{public_pack_path.stem}.backup-{token}.json"
            )
            shutil.copy2(public_pack_path, backup_pack_path)

        shutil.move(str(staging_collection_dir), str(public_collection_dir))
        staging_pack_path.replace(public_pack_path)
    except Exception as error:
        _restore_previous(
            public_collection_dir=public_collection_dir,
            backup_collection_dir=backup_collection_dir,
            public_pack_path=public_pack_path,
            backup_pack_path=backup_pack_path,
        )
        raise LLMWikiRebuildError(f"Failed to publish LLM Wiki: {error}") from error
    finally:
        if backup_collection_dir and backup_collection_dir.exists():
            shutil.rmtree(backup_collection_dir, ignore_errors=True)
        if backup_pack_path and backup_pack_path.exists():
            backup_pack_path.unlink(missing_ok=True)


async def rebuild_llm_wiki(
    collection_id: str,
    *,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_temperature: float | None = None,
    wiki_root: Path | str = DEFAULT_LLM_WIKI_ROOT,
) -> LLMWikiRebuildResponse:
    """Rebuild and publish generated LLM Wiki artifacts for one collection."""
    wiki_root = Path(wiki_root)
    collections_dir = wiki_root / "collections"
    public_collection_dir = collections_dir / collection_id
    public_pack_path = collections_dir / f"{collection_id}.json"
    staging_root = collections_dir / f".staging-{collection_id}-{uuid4().hex}"
    staging_collection_dir = staging_root / collection_id
    staging_pack_path = staging_root / f"{collection_id}.json"
    staging_root.mkdir(parents=True, exist_ok=True)

    try:
        chunks = await _list_all_chunks(collection_id)
        if chunks:
            llm = get_agent_llm(
                provider=llm_provider,
                model=llm_model,
                temperature=llm_temperature,
            )
            source_pages = await _generate_source_pages(
                llm,
                _group_chunks_by_source(chunks),
            )
            concept_pages = await _generate_concept_pages(llm, source_pages)
        else:
            source_pages = []
            concept_pages = []
        generated_at = _now_iso()
        bundle = _ArtifactBundle(
            collection_id=collection_id,
            generated_at=generated_at,
            chunk_count=len(chunks),
            source_pages=source_pages,
            concept_pages=concept_pages,
            public_pack_path=public_pack_path,
        )

        _write_artifacts(
            staging_collection_dir,
            staging_pack_path,
            bundle,
        )
        if not _validate_runtime_pack(staging_pack_path, collection_id):
            raise LLMWikiRebuildError("Staged runtime pack failed schema validation")

        _publish_artifacts(
            staging_collection_dir=staging_collection_dir,
            staging_pack_path=staging_pack_path,
            public_collection_dir=public_collection_dir,
            public_pack_path=public_pack_path,
        )
        return LLMWikiRebuildResponse(
            collection_id=collection_id,
            source_page_count=len(source_pages),
            concept_page_count=len(concept_pages),
            page_count=len(source_pages) + len(concept_pages),
            chunk_count=len(chunks),
            pack_path=str(public_pack_path),
        )
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
