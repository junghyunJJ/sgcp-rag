from __future__ import annotations

import re
from collections import Counter

PDF_MARKDOWN_CLEANUP_VERSION = "minimal_v2"

_STANDALONE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_BIORXIV_BOILERPLATE_PATTERNS = (
    re.compile(r"biorxiv preprint doi:", re.IGNORECASE),
    re.compile(r"not certified by peer review", re.IGNORECASE),
    re.compile(r"author/funder.*display the preprint", re.IGNORECASE),
    re.compile(r"made available under a cc-", re.IGNORECASE),
    re.compile(r"creativecommons\.org/licenses/", re.IGNORECASE),
)
_REPEATED_LINE_MIN_LENGTH = 40
_REPEATED_LINE_MIN_COUNT = 3
_REPEATED_PAGE_FURNITURE_PATTERNS = (
    re.compile(r"\bdoi\.org/", re.IGNORECASE),
    re.compile(r"\bplease cite this article\b", re.IGNORECASE),
    re.compile(r"\bwww\.nature\.com/", re.IGNORECASE),
)


def clean_pdf_markdown(markdown: str) -> str:
    """Remove conservative, high-confidence PDF boilerplate from markdown."""
    lines = markdown.splitlines()
    repeated_lines = _repeated_long_lines(lines)
    cleaned = [
        line
        for line in lines
        if not _should_remove_line(line, repeated_lines=repeated_lines)
    ]
    return _collapse_excess_blank_lines("\n".join(cleaned)).strip() + "\n"


def _repeated_long_lines(lines: list[str]) -> set[str]:
    normalized_lines = [
        normalized
        for line in lines
        if (normalized := _normalize_line(line))
        and len(normalized) >= _REPEATED_LINE_MIN_LENGTH
        and not _TABLE_ROW_RE.match(line)
        and _is_repeated_page_furniture_candidate(line)
    ]
    counts = Counter(normalized_lines)
    return {
        line
        for line, count in counts.items()
        if count >= _REPEATED_LINE_MIN_COUNT
    }


def _should_remove_line(line: str, *, repeated_lines: set[str]) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _STANDALONE_NUMBER_RE.match(stripped):
        return True
    if any(pattern.search(stripped) for pattern in _BIORXIV_BOILERPLATE_PATTERNS):
        return True
    return _normalize_line(line) in repeated_lines


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _is_repeated_page_furniture_candidate(line: str) -> bool:
    return any(pattern.search(line) for pattern in _REPEATED_PAGE_FURNITURE_PATTERNS)


def _collapse_excess_blank_lines(markdown: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", markdown)
