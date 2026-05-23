from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.documents.base import Blob

from langconnect.parsers.pymupdf_parser import PyMuPDF4LLMParser
from langconnect.services.pdf_markdown_quality import (
    MarkdownMetrics,
    compute_markdown_metrics,
    failed_markdown_metrics,
    summarize_parser_metrics,
)

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_PILOT_PDF_DIR = (
    "benchmarking/data/lab-bench/LitQA2-pilot-20-seed42/open_access_fulltext/pdf"
)
DEFAULT_OUTPUT_ROOT = "reports/pdf_markdown_parser_comparison"
EXPECTED_PDF_COUNT = 20


def resolve_repo_root() -> Path:
    """Resolve the repository root from this script location."""
    return Path(__file__).resolve().parents[1]


def collect_pdfs(pdf_dir: Path) -> list[Path]:
    """Return sorted PDF files from a directory."""
    return sorted(path for path in pdf_dir.glob("*.pdf") if path.is_file())


def safe_markdown_name(pdf_path: Path) -> str:
    """Return a filesystem-safe markdown filename for one PDF."""
    safe_stem = "".join(
        char if char.isalnum() or char in "._-" else "_" for char in pdf_path.stem
    )
    return f"{safe_stem}.md"


def parse_with_pymupdf4llm(pdf_path: Path) -> str:
    """Convert one PDF to markdown using the app's current parser stack."""
    blob = Blob(data=pdf_path.read_bytes(), source=str(pdf_path))
    docs = list(PyMuPDF4LLMParser(clean_markdown=True).lazy_parse(blob))
    if not docs:
        raise RuntimeError("PyMuPDF4LLMParser did not return any documents")
    return "\n\n".join(doc.page_content for doc in docs)


def build_marker_converter(*, pdftext_workers: int, disable_ocr: bool):
    """Build a Marker PDF converter lazily so Marker remains optional."""
    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    config_parser = ConfigParser(
        {
            "output_format": "markdown",
            "pdftext_workers": pdftext_workers,
            "disable_ocr": disable_ocr,
        }
    )
    return PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )


def marker_converter_to_callable(converter) -> Callable[[Path], str]:
    """Wrap Marker's rendered output as a PDF path -> markdown callable."""
    from marker.output import text_from_rendered

    def parse(pdf_path: Path) -> str:
        rendered = converter(str(pdf_path))
        markdown, output_format, _images = text_from_rendered(rendered)
        if output_format != "md":
            raise ValueError(f"expected Marker markdown, got {output_format}")
        return markdown

    return parse


def run_parser(
    *,
    parser_name: str,
    pdf_path: Path,
    output_dir: Path,
    parse: Callable[[Path], str],
) -> MarkdownMetrics:
    """Run one parser against one PDF and persist markdown output."""
    started = time.perf_counter()
    try:
        markdown = parse(pdf_path)
        elapsed = time.perf_counter() - started
        output_path = output_dir / safe_markdown_name(pdf_path)
        output_path.write_text(markdown, encoding="utf-8")
        return compute_markdown_metrics(
            markdown,
            parser=parser_name,
            pdf_name=pdf_path.name,
            elapsed_seconds=round(elapsed, 3),
            output_path=str(output_path),
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return failed_markdown_metrics(
            parser=parser_name,
            pdf_name=pdf_path.name,
            elapsed_seconds=round(elapsed, 3),
            error=f"{type(exc).__name__}: {exc}",
        )


def write_csv(metrics: list[MarkdownMetrics], csv_path: Path) -> None:
    """Write flat per-parser metrics for spreadsheet inspection."""
    fieldnames = list(metrics[0].to_dict()) if metrics else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric.to_dict())


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compare PyMuPDF4LLM and Marker markdown on pilot PDFs.",
    )
    parser.add_argument("--pdf-dir", default=DEFAULT_PILOT_PDF_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--parser",
        choices=("all", "pymupdf4llm", "marker"),
        default="all",
    )
    parser.add_argument(
        "--marker-pdftext-workers",
        type=int,
        default=1,
        help="Marker/pdftext worker count. Use 1 for macOS-safe runs.",
    )
    parser.add_argument(
        "--marker-enable-ocr",
        action="store_true",
        help="Enable Marker's OCR pass. Default keeps OCR disabled for text-born PDFs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the parser comparison."""
    args = build_arg_parser().parse_args(argv)
    repo_root = resolve_repo_root()
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.is_absolute():
        pdf_dir = repo_root / pdf_dir
    pdf_dir = pdf_dir.resolve()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    pdfs = collect_pdfs(pdf_dir)
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        print(f"no PDFs found in {pdf_dir}", file=sys.stderr)
        return 2

    if args.limit is None and len(pdfs) != EXPECTED_PDF_COUNT:
        print(
            f"warning: expected {EXPECTED_PDF_COUNT} PDFs, found {len(pdfs)}",
            file=sys.stderr,
        )

    parser_calls: list[tuple[str, Callable[[Path], str]]] = []
    marker_init_seconds: float | None = None
    if args.parser in {"all", "pymupdf4llm"}:
        parser_calls.append(("pymupdf4llm", parse_with_pymupdf4llm))
    if args.parser in {"all", "marker"}:
        started = time.perf_counter()
        marker_converter = build_marker_converter(
            pdftext_workers=args.marker_pdftext_workers,
            disable_ocr=not args.marker_enable_ocr,
        )
        marker_init_seconds = round(time.perf_counter() - started, 3)
        parser_calls.append(("marker", marker_converter_to_callable(marker_converter)))

    metrics: list[MarkdownMetrics] = []
    for index, pdf_path in enumerate(pdfs, 1):
        print(f"pdf={index}/{len(pdfs)} file={pdf_path.name}", flush=True)
        for parser_name, parse in parser_calls:
            parser_output_dir = output_root / parser_name
            parser_output_dir.mkdir(parents=True, exist_ok=True)
            metric = run_parser(
                parser_name=parser_name,
                pdf_path=pdf_path,
                output_dir=parser_output_dir,
                parse=parse,
            )
            metrics.append(metric)
            status = "ok" if metric.ok else "failed"
            print(
                f"  parser={parser_name} status={status} "
                f"seconds={metric.elapsed_seconds} chars={metric.char_count}",
                flush=True,
            )

    summary = {
        "pdf_dir": str(pdf_dir),
        "output_root": str(output_root),
        "pdf_count": len(pdfs),
        "marker_init_seconds": marker_init_seconds,
        "marker_config": {
            "pdftext_workers": args.marker_pdftext_workers,
            "disable_ocr": not args.marker_enable_ocr,
        }
        if args.parser in {"all", "marker"}
        else None,
        "summary": summarize_parser_metrics(metrics),
        "results": [metric.to_dict() for metric in metrics],
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(metrics, output_root / "results.csv")

    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))
    print(f"wrote={summary_path}")
    return 0 if all(metric.ok for metric in metrics) else 1


if __name__ == "__main__":
    raise SystemExit(main())
