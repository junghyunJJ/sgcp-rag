from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from langconnect.database.collections import Collection, CollectionsManager
from langconnect.database.connection import close_db_pool, get_db_connection
from langconnect.services.document_processor import process_document
from langconnect.services.paper_cards import repo_relative_path, resolve_repo_root


DEFAULT_COLLECTION_NAME = "litqa2-full"
DEFAULT_PDF_DIR = "benchmarking/data/lab-bench/LitQA2/open_access_fulltext/pdf"
DEFAULT_REPORT_PATH = "reports/litqa2-full-ingest-status.jsonl"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


class LocalUploadFile:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.filename = path.name
        self.content_type = "application/pdf"

    async def read(self) -> bytes:
        return self.path.read_bytes()


def _json_default(value: object) -> str:
    return str(value)


def _append_status(report_path: Path, payload: dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=_json_default, sort_keys=True) + "\n")


async def _get_or_create_collection(
    *,
    name: str,
    pdf_dir: Path,
    expected_pdf_count: int,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    manager = CollectionsManager()
    for collection in await manager.list():
        if collection["name"] == name:
            return collection["uuid"]

    created = await manager.create(
        name,
        {
            "dataset": "LitQA2",
            "source_dir": repo_relative_path(pdf_dir),
            "pdf_count": expected_pdf_count,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "llm_wiki_rebuild": "skipped_by_request",
            "created_for": "lab-bench llmwiki full ingest",
            "created_at": datetime.now().isoformat(),
        },
    )
    if not created:
        raise RuntimeError(f"Failed to create collection {name!r}")
    return created["uuid"]


async def _existing_source_paths(collection_id: str) -> set[str]:
    async with get_db_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT e.cmetadata->>'source_path' AS source_path
              FROM langchain_pg_embedding e
             WHERE e.collection_id = $1
               AND e.cmetadata ? 'source_path'
            """,
            collection_id,
        )
    return {row["source_path"] for row in rows if row["source_path"]}


async def _collection_counts(collection_id: str) -> dict[str, int]:
    async with get_db_connection() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                COUNT(*)::int AS chunk_count,
                COUNT(DISTINCT e.cmetadata->>'source_path')::int AS source_path_count,
                COUNT(DISTINCT e.cmetadata->>'file_id')::int AS file_id_count
              FROM langchain_pg_embedding e
             WHERE e.collection_id = $1
            """,
            collection_id,
        )
    if row is None:
        return {"chunk_count": 0, "source_path_count": 0, "file_id_count": 0}
    return {
        "chunk_count": row["chunk_count"],
        "source_path_count": row["source_path_count"],
        "file_id_count": row["file_id_count"],
    }


async def ingest(args: argparse.Namespace) -> int:
    repo_root = resolve_repo_root()
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.is_absolute():
        pdf_dir = repo_root / pdf_dir
    pdf_dir = pdf_dir.resolve()
    report_path = Path(args.report_path)
    if not report_path.is_absolute():
        report_path = repo_root / report_path

    pdfs = sorted(path for path in pdf_dir.glob("*.pdf") if path.is_file())
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}", file=sys.stderr)
        return 2

    collection_id = await _get_or_create_collection(
        name=args.collection_name,
        pdf_dir=pdf_dir,
        expected_pdf_count=len(pdfs),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    collection = Collection(collection_id=collection_id)
    existing_sources = await _existing_source_paths(collection_id)

    print(
        "ingest_start "
        f"collection_name={args.collection_name} collection_id={collection_id} "
        f"pdfs={len(pdfs)} existing_sources={len(existing_sources)} "
        f"report={report_path}",
        flush=True,
    )

    processed = 0
    skipped = 0
    failed = 0
    for index, path in enumerate(pdfs, 1):
        source_path = repo_relative_path(path, repo_root)
        if source_path in existing_sources:
            skipped += 1
            print(f"skip={index}/{len(pdfs)} file={path.name}", flush=True)
            continue

        metadata = {
            "source": path.name,
            "created_at": datetime.now().isoformat(),
            "filename": path.name,
            "mime_type": "application/pdf",
        }
        if source_path:
            metadata["source_path"] = source_path

        try:
            print(f"process={index}/{len(pdfs)} file={path.name}", flush=True)
            docs = await process_document(
                LocalUploadFile(path),
                metadata=metadata,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                collection_id=collection_id,
            )
            added_ids = await collection.upsert(docs)
            processed += 1
            if source_path:
                existing_sources.add(source_path)
            _append_status(
                report_path,
                {
                    "status": "success",
                    "collection_id": collection_id,
                    "source_path": source_path,
                    "filename": path.name,
                    "chunks": len(added_ids),
                    "created_at": datetime.now().isoformat(),
                },
            )
            print(
                f"done={index}/{len(pdfs)} file={path.name} chunks={len(added_ids)}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            _append_status(
                report_path,
                {
                    "status": "failed",
                    "collection_id": collection_id,
                    "source_path": source_path,
                    "filename": path.name,
                    "error": repr(exc),
                    "created_at": datetime.now().isoformat(),
                },
            )
            print(f"FAIL {path.name}: {exc!r}", file=sys.stderr, flush=True)
            if args.fail_fast:
                raise

    counts = await _collection_counts(collection_id)
    card_dir = repo_root / "llm_wiki" / "paper_cards" / collection_id
    card_count = len(list(card_dir.glob("*.json"))) if card_dir.exists() else 0
    summary = {
        "status": "summary",
        "collection_name": args.collection_name,
        "collection_id": collection_id,
        "pdf_count": len(pdfs),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "paper_card_count": card_count,
        **counts,
        "created_at": datetime.now().isoformat(),
    }
    _append_status(report_path, summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest full LitQA2 PDFs and paper cards without rebuilding LLM Wiki.",
    )
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--pdf-dir", default=DEFAULT_PDF_DIR)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return await ingest(args)
    finally:
        await close_db_pool()


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
