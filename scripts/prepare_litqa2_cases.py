"""Prepare LitQA2 JSONL cases for the wiki off/on benchmark runner."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_QUESTION_TYPE = "litqa2"
DOI_URL_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
)
HTML_TAG_RE = re.compile(r"<[^>]+>")


class LitQA2ConfigError(RuntimeError):
    """Raised when LitQA2 inputs or options are invalid."""


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LitQA2ConfigError(f"Invalid JSON file: {path}") from exc


def load_litqa2_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load LitQA2 public JSONL records."""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LitQA2ConfigError(
                f"Invalid LitQA2 JSONL at {path}:{line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise LitQA2ConfigError(
                f"LitQA2 record at {path}:{line_number} is not an object"
            )
        records.append(record)
    return records


def normalize_doi(value: object) -> str | None:
    """Return a lowercase DOI string without URL prefixes."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    lowered = text.lower()
    for prefix in DOI_URL_PREFIXES:
        if lowered.startswith(prefix):
            return text[len(prefix) :].strip().lower()
    if lowered.startswith("doi:"):
        return text[4:].strip().lower()
    if lowered.startswith("10."):
        return lowered
    return None


def _doi_url(doi: str) -> str:
    return f"https://doi.org/{doi}"


def _dedupe_texts(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = value.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _clean_title(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    title = html.unescape(HTML_TAG_RE.sub("", value))
    title = " ".join(title.split())
    return title or None


def _text_or_none(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _source_values(record: dict[str, Any]) -> list[str]:
    sources: list[str] = []
    raw_sources = record.get("sources")
    if isinstance(raw_sources, list):
        sources.extend(item.strip() for item in raw_sources if _text_or_none(item))
    source = _text_or_none(record.get("source"))
    if source is not None:
        sources.append(source)
    return _dedupe_texts(sources)


def _manifest_aliases(record: dict[str, Any]) -> list[str]:
    aliases: list[str] = []
    for key in ("doi", "source_url", "saved_path", "download_url"):
        value = _text_or_none(record.get(key))
        if value is not None:
            aliases.append(value)
            aliases.append(value.lower())
    doi = normalize_doi(record.get("doi")) or normalize_doi(record.get("source_url"))
    if doi is not None:
        aliases.extend([doi, _doi_url(doi)])
    saved_path = _text_or_none(record.get("saved_path"))
    if saved_path is not None:
        aliases.append(Path(saved_path).name)
    return _dedupe_texts(aliases)


def load_manifest_index(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load the LitQA2 fulltext manifest as a lookup index."""
    if path is None:
        return {}
    data = _read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise LitQA2ConfigError("LitQA2 manifest must contain a records list")
    index: dict[str, dict[str, Any]] = {}
    for record in data["records"]:
        if not isinstance(record, dict):
            continue
        for alias in _manifest_aliases(record):
            index.setdefault(alias, record)
            index.setdefault(alias.lower(), record)
    return index


def _manifest_for_source(
    source: str,
    manifest_index: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    doi = normalize_doi(source)
    aliases = [source, source.lower()]
    if doi is not None:
        aliases.extend([doi, _doi_url(doi)])
    for alias in aliases:
        record = manifest_index.get(alias) or manifest_index.get(alias.lower())
        if record is not None:
            return record
    return None


def _evidence_key(source: str, manifest_record: dict[str, Any] | None) -> str:
    if manifest_record is not None:
        saved_path = _text_or_none(manifest_record.get("saved_path"))
        if saved_path is not None:
            return Path(saved_path).name
    return source


def _litqa2_metadata(
    record: dict[str, Any],
    sources: list[str],
    manifest_records: list[dict[str, Any]],
) -> dict[str, Any]:
    dois: list[str] = []
    saved_paths: list[str] = []
    titles: list[str] = []
    for source in sources:
        doi = normalize_doi(source)
        if doi is not None:
            dois.append(doi)
    for manifest_record in manifest_records:
        doi = normalize_doi(manifest_record.get("doi"))
        if doi is not None:
            dois.append(doi)
        saved_path = _text_or_none(manifest_record.get("saved_path"))
        if saved_path is not None:
            saved_paths.append(saved_path)
        openalex = manifest_record.get("openalex")
        if isinstance(openalex, dict):
            title = _clean_title(openalex.get("title"))
            if title is not None:
                titles.append(title)

    metadata: dict[str, Any] = {
        "is_opensource": record.get("is_opensource") is True,
        "sources": sources,
    }
    if dois:
        metadata["doi"] = _dedupe_texts(dois)
    if saved_paths:
        metadata["saved_paths"] = _dedupe_texts(saved_paths)
    if titles:
        metadata["titles"] = _dedupe_texts(titles)
    distractors = record.get("distractors")
    if isinstance(distractors, list):
        metadata["distractors"] = distractors
    for key in ("tag", "version"):
        value = record.get(key)
        if isinstance(value, str | int | float | bool):
            metadata[key] = str(value)
    return metadata


def _require_text(record: dict[str, Any], key: str, *, case_id: str) -> str:
    value = _text_or_none(record.get(key))
    if value is None:
        raise LitQA2ConfigError(f"LitQA2 case {case_id!r} is missing {key!r}")
    return value


def _question_type(record: dict[str, Any]) -> str:
    value = _text_or_none(record.get("subtask"))
    return value or DEFAULT_QUESTION_TYPE


def convert_litqa2_cases(
    records: list[dict[str, Any]],
    *,
    manifest_index: dict[str, dict[str, Any]] | None = None,
    open_source_only: bool = True,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Convert LitQA2 records to the JSON schema accepted by the benchmark runner."""
    if limit is not None and limit < 1:
        raise LitQA2ConfigError("--limit must be at least 1")
    manifest_index = manifest_index or {}
    cases: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if open_source_only and record.get("is_opensource") is not True:
            continue
        case_id = str(record.get("id") or f"litqa2-{index:04d}")
        sources = _source_values(record)
        manifest_records = [
            manifest_record
            for manifest_record in (
                _manifest_for_source(source, manifest_index) for source in sources
            )
            if manifest_record is not None
        ]
        fact = _text_or_none(record.get("key-passage"))
        evidence_list: list[dict[str, str]] = []
        for source in sources:
            evidence_key = _evidence_key(
                source,
                _manifest_for_source(source, manifest_index),
            )
            evidence = {"source": evidence_key, "document_id": evidence_key}
            if fact is not None:
                evidence["fact"] = fact
            if evidence not in evidence_list:
                evidence_list.append(evidence)

        cases.append(
            {
                "id": case_id,
                "query": _require_text(record, "question", case_id=case_id),
                "answer": _require_text(record, "ideal", case_id=case_id),
                "question_type": _question_type(record),
                "evidence_list": evidence_list,
                "litqa2": _litqa2_metadata(record, sources, manifest_records),
            }
        )
        if limit is not None and len(cases) >= limit:
            break
    return cases


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LitQA2 public JSONL into benchmark case JSON.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--include-closed-source",
        action="store_true",
        help="Include records where is_opensource is false.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    try:
        cases = convert_litqa2_cases(
            load_litqa2_jsonl(args.input),
            manifest_index=load_manifest_index(args.manifest),
            open_source_only=not args.include_closed_source,
            limit=args.limit,
        )
    except LitQA2ConfigError as exc:
        print(f"litqa2 prepare error: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(cases)} LitQA2 benchmark cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
