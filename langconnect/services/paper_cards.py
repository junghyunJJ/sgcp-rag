from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from langconnect.models.paper_card import PaperCardExtractionQuality, PaperCardV0

COLLECTION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
MIN_TITLE_CHARS = 8
TITLE_CHAR_LIMIT = 300
MIN_ABSTRACT_CHARS = 300
MIN_SENTENCE_LIKE_SPANS = 2
MIN_PROSE_LINE_CHARS = 40
MIN_PROSE_LINE_WORDS = 6
AUTHOR_LINE_COMMA_LIMIT = 4
MIN_AUTHOR_NAME_CHUNKS = 2
MIN_AUTHOR_COMMAS = 2
KEYWORD_PIPE_LIMIT = 2
PARAGRAPH_BREAK_BLANK_LINES = 1
STRONG_PARAGRAPH_BREAK_BLANK_LINES = 2
MIN_CONTINUATION_CHARS = 3
MIN_CONTINUATION_WORDS = 2
ABSTRACT_START_HEADINGS = {"abstract", "summary"}
ABSTRACT_STOP_HEADINGS = {
    "keywords",
    "keyword",
    "introduction",
    "background",
    "results",
    "methods",
    "materials and methods",
    "main",
    "graphical abstract",
    "highlights",
    "significance",
    "author summary",
    "simple summary",
    "lay summary",
}
TITLE_SKIP_HEADINGS = {"abstract", "introduction", "keywords", "keyword"}
ARTICLE_TYPE_HEADINGS = {"article", "articles", "research article", "resource", "investigation", "editorial"}
PROSE_STOP_PATTERNS = (
    " department ",
    " university",
    " institute",
    " correspondence",
    " nature |",
    " nature communications |",
    " author contributions",
    " reviewers:",
    " journal homepage",
    " therapeutics, san francisco",
)
PROSE_SKIP_PATTERNS = (
    "http://",
    "https://",
    "doi.org",
    "mailto:",
    "received:",
    "accepted:",
    "published online",
    "open access",
    "check for updates",
    "article history",
    "advance access publication date",
    "to cite this article",
    "to link to this article",
    "full terms & conditions",
)
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
URL_RE = re.compile(r"https?://\S+")
NAME_CHUNK_RE = re.compile(
    r"\b(?:[A-Z][a-zA-ZÀ-ÿ.-]+|[A-Z]\.)\s+"
    r"(?:[A-Z]\.\s+)?[A-Z][a-zA-ZÀ-ÿ.-]+\b"
)


class PaperCardPathError(ValueError):
    """Raised when a paper-card artifact path would be unsafe."""


def resolve_repo_root(start: Path | None = None) -> Path:
    """Resolve the repository root from a starting path or this module path."""
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "langconnect"
        ).is_dir():
            return candidate

    return Path(__file__).resolve().parents[2]


DEFAULT_PAPER_CARD_ROOT = resolve_repo_root() / "llm_wiki"


def paper_card_root() -> Path:
    """Return the default paper-card output root under the repository."""
    return DEFAULT_PAPER_CARD_ROOT


def content_hash(pdf_bytes: bytes) -> str:
    """Return a stable sha256 digest string for PDF bytes."""
    return "sha256:" + hashlib.sha256(pdf_bytes).hexdigest()


def slugify_source(source: str) -> str:
    """Convert a source filename or path into a filesystem-safe slug."""
    stem = Path(source).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return slug or "paper"


def paper_card_filename(source: str, digest: str) -> str:
    """Build the deterministic paper-card filename for a source and digest."""
    prefix = digest.split(":", 1)[-1][:12]
    return f"{slugify_source(source)}--sha256-{prefix}.json"


def _safe_child_path(base: Path, *parts: str) -> Path:
    resolved_base = base.resolve()
    path = resolved_base.joinpath(*parts).resolve()
    if not path.is_relative_to(resolved_base):
        raise PaperCardPathError("Unsafe paper-card artifact path")
    return path


def _validate_collection_id(collection_id: str) -> str:
    text = str(collection_id).strip()
    if (
        not text
        or "/" in text
        or "\\" in text
        or ".." in text
        or not COLLECTION_ID_RE.fullmatch(text)
    ):
        raise PaperCardPathError("Invalid paper-card collection identifier")
    return text


def repo_relative_path(
    path: str | Path | None,
    repo_root: str | Path | None = None,
) -> str | None:
    """Return a repo-relative path if the input resolves inside the repo."""
    if path is None:
        return None

    root = Path(repo_root).expanduser().resolve() if repo_root else resolve_repo_root()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        relative = candidate.resolve().relative_to(root)
    except ValueError:
        return None
    return sanitize_source_path(relative.as_posix(), root)


def sanitize_source_path(  # noqa: PLR0911
    value: str | Path | None,
    repo_root: Path | None = None,
) -> str | None:
    """Keep only safe repo-relative source paths for persisted card JSON."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None
    if "\\" in text or re.match(r"^[A-Za-z]:", text):
        return None

    raw = Path(text).expanduser()
    if raw.is_absolute():
        return None

    root = repo_root.resolve() if repo_root else resolve_repo_root()
    try:
        resolved = (root / raw).resolve()
        relative = resolved.relative_to(root)
    except ValueError:
        return None

    if any(part == ".." for part in relative.parts):
        return None
    return relative.as_posix()


def paper_card_collection_dir(
    collection_id: str,
    root: Path | str | None = None,
) -> Path:
    """Return a safe output directory for one collection's paper cards."""
    safe_collection_id = _validate_collection_id(collection_id)
    output_root = Path(root).expanduser() if root is not None else paper_card_root()
    paper_cards_dir = output_root.resolve() / "paper_cards"
    return _safe_child_path(paper_cards_dir, safe_collection_id)


def _is_markdown_heading(line: str) -> bool:
    return bool(re.match(r"^\s{0,3}#{1,6}\s+", line))


def _clean_line(line: str) -> str:
    text = line.replace("\u00ad", "")
    previous = None
    while previous != text:
        previous = text
        text = LINK_RE.sub(r"\1", text)
    text = URL_RE.sub("", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text).strip()
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", text)
    text = re.sub(r"^\s*\d{1,4}\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -*\t")


def _heading_text(line: str) -> str | None:
    text = _clean_line(line).lower().rstrip(":")
    text = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", text)
    return text or None


def _extract_title(markdown: str, source: str) -> tuple[str | None, list[str]]:
    for line in markdown.splitlines():
        text = _clean_line(line)
        if not text:
            continue
        if text.lower().rstrip(":") in TITLE_SKIP_HEADINGS:
            continue
        if len(text) >= MIN_TITLE_CHARS:
            return text[:TITLE_CHAR_LIMIT], []
    return Path(source).stem, ["title_not_found"]


def _sentence_like_spans(text: str) -> int:
    return len(re.findall(r"[^.!?]{20,}[.!?]", text))


def _quality_warnings(abstract: str, spans: int) -> list[str]:
    warnings: list[str] = []
    if len(abstract) < MIN_ABSTRACT_CHARS:
        warnings.append("abstract_short")
    if spans < MIN_SENTENCE_LIKE_SPANS:
        warnings.append("abstract_not_sentence_like")
    return warnings


def _inline_abstract_start(line: str) -> tuple[str, str] | None:
    match = re.match(
        r"^(abstract|summary)\s*:\s*(.*)$",
        _clean_line(line),
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return match.group(1).lower(), match.group(2).strip()


def _is_keyword_line(text: str) -> bool:
    lower = text.lower().strip()
    if re.match(r"^(keywords?|key words?)\b\s*[:;]?", lower):
        return True
    return text.count("|") >= KEYWORD_PIPE_LIMIT and len(text) < TITLE_CHAR_LIMIT


def _extract_heading_abstract(markdown: str) -> tuple[str | None, list[str], int]:
    lines = markdown.splitlines()
    start: int | None = None
    first_text = ""
    for index, line in enumerate(lines):
        inline_start = _inline_abstract_start(line)
        if inline_start is not None:
            label, first_text = inline_start
            if label == "abstract":
                start = index + 1
                break
        if _heading_text(line) in ABSTRACT_START_HEADINGS:
            start = index + 1
            break

    if start is None:
        return None, ["abstract_not_found"], 0

    collected: list[str] = []
    if first_text:
        collected.append(first_text)
    for line in lines[start:]:
        heading = _heading_text(line)
        cleaned = _clean_line(line)
        if heading in ABSTRACT_STOP_HEADINGS or _is_keyword_line(cleaned):
            break
        if cleaned:
            collected.append(cleaned)

    abstract = " ".join(collected).strip()
    if not abstract:
        return None, ["abstract_empty"], 0

    spans = _sentence_like_spans(abstract)
    return abstract, _quality_warnings(abstract, spans), spans


def _is_metadata_or_journal_line(text: str) -> bool:
    lower = f" {text.lower()} "
    stripped = text.lower().strip()
    if stripped in ARTICLE_TYPE_HEADINGS:
        return True
    if stripped in {"usa", "canada", "germany", "switzerland", "denmark", "romania", "china", "japan"}:
        return True
    if _is_keyword_line(text):
        return True
    if any(pattern in lower for pattern in PROSE_STOP_PATTERNS):
        return True
    return bool(
        re.search(
            r"\b(Department|University|Institute|Laboratory|School of|Faculty of|Hospital|Company)\b",
            text,
        )
    )


def _is_author_line(text: str) -> bool:
    lower = text.lower()
    if "✉" in text or "email:" in lower or "contributed equally" in lower:
        return True
    name_chunks = len(NAME_CHUNK_RE.findall(text))
    comma_count = text.count(",")
    has_name_joiner = " & " in text or bool(re.search(r"\s+and\s+", text))
    ends_sentence = bool(re.search(r"[.!?]$", text))
    if (
        name_chunks >= MIN_AUTHOR_NAME_CHUNKS
        and not ends_sentence
        and (comma_count >= MIN_AUTHOR_COMMAS or has_name_joiner)
    ):
        return True
    return comma_count >= AUTHOR_LINE_COMMA_LIMIT and not ends_sentence


def _is_candidate_prose_line(text: str) -> bool:
    lower = text.lower()
    if len(text) < MIN_PROSE_LINE_CHARS or len(text.split()) < MIN_PROSE_LINE_WORDS:
        return False
    if any(pattern in lower for pattern in PROSE_SKIP_PATTERNS):
        return False
    if _is_metadata_or_journal_line(text):
        return False
    if _is_author_line(text):
        return False
    return bool(re.search(r"[a-z]", text))


def _is_continuation_prose_line(text: str) -> bool:
    if len(text) < MIN_CONTINUATION_CHARS or len(text.split()) < MIN_CONTINUATION_WORDS:
        return False
    if _is_metadata_or_journal_line(text):
        return False
    return bool(re.search(r"[a-z]", text))


def _should_stop_after_blank(
    collected: list[str],
    next_text: str,
    blank_run: int,
) -> bool:
    if (
        not collected
        or blank_run < PARAGRAPH_BREAK_BLANK_LINES
        or not _is_continuation_prose_line(next_text)
    ):
        return False
    if blank_run >= STRONG_PARAGRAPH_BREAK_BLANK_LINES:
        return True
    current = " ".join(collected).strip()
    return (
        len(current) >= MIN_ABSTRACT_CHARS
        and _sentence_like_spans(current) >= MIN_SENTENCE_LIKE_SPANS
        and bool(re.search(r"[.!?]$", current))
    )


def _extract_unheaded_lead_abstract(
    markdown: str,
) -> tuple[str | None, list[str], int]:
    collected: list[str] = []
    blank_run = 0
    for line in markdown.splitlines():
        cleaned = _clean_line(line)
        if not cleaned:
            blank_run += 1
            continue

        heading = _heading_text(line)
        if heading in ABSTRACT_STOP_HEADINGS:
            return (None, ["abstract_not_found"], 0) if not collected else _finalize_abstract(collected)

        if (
            not collected
            and (_is_markdown_heading(line) or heading in TITLE_SKIP_HEADINGS | ARTICLE_TYPE_HEADINGS)
        ):
            blank_run = 0
            continue
        if (
            collected
            and _should_stop_after_blank(collected, cleaned, blank_run)
        ):
            break
        if collected and _is_metadata_or_journal_line(cleaned):
            break

        if collected:
            if _is_continuation_prose_line(cleaned):
                collected.append(cleaned)
            blank_run = 0
            continue

        if not _is_candidate_prose_line(cleaned):
            blank_run = 0
            continue
        collected.append(cleaned)
        blank_run = 0

    return _finalize_abstract(collected)


def _finalize_abstract(collected: list[str]) -> tuple[str | None, list[str], int]:
    abstract = " ".join(collected).strip()
    if not abstract:
        return None, ["abstract_not_found"], 0
    spans = _sentence_like_spans(abstract)
    return abstract, _quality_warnings(abstract, spans), spans


def _extract_abstract(markdown: str) -> tuple[str | None, list[str], int]:
    abstract, warnings, spans = _extract_heading_abstract(markdown)
    if abstract is not None or "abstract_empty" in warnings:
        return abstract, warnings, spans
    return _extract_unheaded_lead_abstract(markdown)


def build_paper_card_v0(  # noqa: PLR0913
    *,
    collection_id: str,
    markdown: str,
    pdf_bytes: bytes,
    source: str,
    filename: str,
    source_path: str | None,
    parser: str,
    parser_version: str,
) -> PaperCardV0:
    """Build a v0 abstract-only paper card from parsed PDF markdown."""
    digest = content_hash(pdf_bytes)
    title, title_warnings = _extract_title(markdown, source)
    abstract, abstract_warnings, spans = _extract_abstract(markdown)
    warnings = [*title_warnings, *abstract_warnings]
    safe_source_path = sanitize_source_path(source_path)
    quality = PaperCardExtractionQuality(
        has_title=bool(title),
        has_abstract=bool(abstract),
        abstract_chars=len(abstract or ""),
        abstract_sentence_like_spans=spans,
        warnings=warnings,
    )
    return PaperCardV0(
        collection_id=_validate_collection_id(collection_id),
        source=source,
        filename=filename,
        source_path=safe_source_path,
        content_hash=digest,
        parser=parser,
        parser_version=parser_version,
        title=title,
        abstract=abstract,
        extraction_quality=quality,
    )


def write_paper_card(
    card: PaperCardV0,
    *,
    root: Path | str | None = None,
) -> Path:
    """Persist a paper card as JSON and return the written path."""
    base = paper_card_collection_dir(card.collection_id, root)
    base.mkdir(parents=True, exist_ok=True)
    safe_card = card.model_copy(
        update={"source_path": sanitize_source_path(card.source_path)}
    )
    path = _safe_child_path(base, paper_card_filename(card.source, card.content_hash))
    path.write_text(
        json.dumps(safe_card.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return path
