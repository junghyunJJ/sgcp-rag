from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S+", re.MULTILINE)
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_IMAGE_RE = re.compile(r"!\[[^\]]*]\([^)]+\)")
_LINK_RE = re.compile(r"(?<!!)\[[^\]]+]\([^)]+\)")
_MATH_BLOCK_RE = re.compile(r"(^|\n)\s*\$\$.*?\$\$\s*(?=\n|$)", re.DOTALL)
_WORD_RE = re.compile(r"\b[\w'-]+\b")
_ABSTRACT_RE = re.compile(
    r"(^|\n)\s{0,3}#{0,6}\s*\*{0,2}abstract\*{0,2}\b",
    re.IGNORECASE,
)
_REFERENCES_RE = re.compile(
    r"(^|\n)\s{0,3}#{0,6}\s*\*{0,2}(references|bibliography)\*{0,2}\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class MarkdownMetrics:
    parser: str
    pdf_name: str
    ok: bool
    elapsed_seconds: float = 0
    output_path: str | None = None
    error: str | None = None
    char_count: int = 0
    word_count: int = 0
    line_count: int = 0
    heading_count: int = 0
    table_line_count: int = 0
    image_count: int = 0
    link_count: int = 0
    math_block_count: int = 0
    abstract_signal: bool = False
    references_signal: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable metrics dictionary."""
        return asdict(self)


def compute_markdown_metrics(
    markdown: str,
    *,
    parser: str = "",
    pdf_name: str = "",
    elapsed_seconds: float = 0,
    output_path: str | None = None,
) -> MarkdownMetrics:
    """Compute lightweight RAG-relevant markdown quality proxy metrics."""
    return MarkdownMetrics(
        parser=parser,
        pdf_name=pdf_name,
        ok=True,
        elapsed_seconds=elapsed_seconds,
        output_path=output_path,
        char_count=len(markdown),
        word_count=len(_WORD_RE.findall(markdown)),
        line_count=len(markdown.splitlines()),
        heading_count=len(_HEADING_RE.findall(markdown)),
        table_line_count=len(_TABLE_LINE_RE.findall(markdown)),
        image_count=len(_IMAGE_RE.findall(markdown)),
        link_count=len(_LINK_RE.findall(markdown)),
        math_block_count=len(_MATH_BLOCK_RE.findall(markdown)),
        abstract_signal=bool(_ABSTRACT_RE.search(markdown)),
        references_signal=bool(_REFERENCES_RE.search(markdown)),
    )


def failed_markdown_metrics(
    *,
    parser: str,
    pdf_name: str,
    elapsed_seconds: float,
    error: str,
) -> MarkdownMetrics:
    """Create a metrics record for a failed parser run."""
    return MarkdownMetrics(
        parser=parser,
        pdf_name=pdf_name,
        ok=False,
        elapsed_seconds=elapsed_seconds,
        error=error,
    )


def summarize_parser_metrics(
    metrics: list[MarkdownMetrics],
) -> dict[str, dict[str, float | int]]:
    """Aggregate per-parser benchmark metrics."""
    parser_names = sorted({metric.parser for metric in metrics})
    summary: dict[str, dict[str, float | int]] = {}
    for parser in parser_names:
        parser_metrics = [metric for metric in metrics if metric.parser == parser]
        successful = [metric for metric in parser_metrics if metric.ok]
        summary[parser] = {
            "attempted": len(parser_metrics),
            "succeeded": len(successful),
            "failed": len(parser_metrics) - len(successful),
            "mean_elapsed_seconds": round(
                mean(metric.elapsed_seconds for metric in successful), 3
            )
            if successful
            else 0,
            "mean_chars": round(mean(metric.char_count for metric in successful))
            if successful
            else 0,
            "mean_words": round(mean(metric.word_count for metric in successful))
            if successful
            else 0,
            "mean_headings": round(
                mean(metric.heading_count for metric in successful), 1
            )
            if successful
            else 0,
            "mean_table_lines": round(
                mean(metric.table_line_count for metric in successful), 1
            )
            if successful
            else 0,
            "mean_images": round(mean(metric.image_count for metric in successful), 1)
            if successful
            else 0,
            "abstract_signal_count": sum(
                metric.abstract_signal for metric in successful
            ),
            "references_signal_count": sum(
                metric.references_signal for metric in successful
            ),
        }
    return summary
