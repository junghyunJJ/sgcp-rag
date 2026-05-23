from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from langchain_core.documents.base import Blob

from langconnect.parsers.pymupdf_parser import PyMuPDF4LLMParser
from langconnect.services.paper_cards import (
    build_paper_card_v0,
    repo_relative_path,
    resolve_repo_root,
    write_paper_card,
)

DEFAULT_PILOT_PDF_DIR = (
    "benchmarking/data/lab-bench/LitQA2-pilot-20-seed42/open_access_fulltext/pdf"
)
SMOKE_COLLECTION_ID = "smoke-litqa2-pilot-20"
EXPECTED_PDF_COUNT = 20


def pdf_to_markdown(path: Path) -> str:
    """Convert one PDF file to markdown using the app's PDF stack."""
    blob = Blob(data=path.read_bytes(), source=str(path))
    docs = list(PyMuPDF4LLMParser(clean_markdown=True).lazy_parse(blob))
    if not docs:
        raise RuntimeError("PyMuPDF4LLMParser did not return any documents")
    return "\n\n".join(doc.page_content for doc in docs)


def collect_pdfs(pdf_dir: Path) -> list[Path]:
    """Return sorted PDF files from a directory."""
    return sorted(path for path in pdf_dir.glob("*.pdf") if path.is_file())


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for the smoke script."""
    parser = argparse.ArgumentParser(
        description="Smoke test abstract paper cards on the LitQA2 pilot PDFs.",
    )
    parser.add_argument("--pdf-dir", default=DEFAULT_PILOT_PDF_DIR)
    parser.add_argument("--output-root")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the LitQA2 pilot paper-card smoke test."""
    args = build_arg_parser().parse_args(argv)
    repo_root = resolve_repo_root()
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.is_absolute():
        pdf_dir = repo_root / pdf_dir
    pdf_dir = pdf_dir.resolve()

    pdfs = collect_pdfs(pdf_dir)
    if len(pdfs) != EXPECTED_PDF_COUNT:
        print(
            f"expected {EXPECTED_PDF_COUNT} PDFs, found {len(pdfs)} in {pdf_dir}",
            file=sys.stderr,
        )
        return 2

    output_root = Path(args.output_root) if args.output_root else Path(
        tempfile.mkdtemp(prefix="paper-card-smoke-")
    )
    if not output_root.is_absolute():
        output_root = repo_root / output_root

    processed = 0
    failures = 0
    for index, path in enumerate(pdfs, 1):
        try:
            print(f"processing={index}/{EXPECTED_PDF_COUNT} file={path.name}", flush=True)
            pdf_bytes = path.read_bytes()
            markdown = pdf_to_markdown(path)
            card = build_paper_card_v0(
                collection_id=SMOKE_COLLECTION_ID,
                markdown=markdown,
                pdf_bytes=pdf_bytes,
                source=path.name,
                filename=path.name,
                source_path=repo_relative_path(path, repo_root),
                parser="PyMuPDF4LLMParser",
                parser_version="pymupdf4llm",
            )
            write_paper_card(card, root=output_root)
            processed += 1
        except Exception as exc:
            failures += 1
            print(f"FAIL {path.name}: {exc}", file=sys.stderr)

    card_dir = output_root / "paper_cards" / SMOKE_COLLECTION_ID
    cards = sorted(card_dir.glob("*.json")) if card_dir.exists() else []
    print(
        f"processed={processed} cards={len(cards)} failures={failures} "
        f"output_root={output_root}"
    )
    if processed != EXPECTED_PDF_COUNT or len(cards) != EXPECTED_PDF_COUNT or failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
