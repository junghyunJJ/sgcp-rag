"""Ingest MultiHop-RAG news articles (.txt) into a collection.

Reads benchmarking/data/multihoprag/articles_manifest.json and ingests each
article as full text (chunk_size 3000 / overlap 200), attaching source/title/url
metadata so the benchmark's evidence_recall can match retrieved chunks to the
MultiHopRAG.json evidence_list entries. No LLM Wiki rebuild here.

Run inside the API container:

    docker cp scripts/ingest_multihop.py langconnect-api:/app/scripts/
    docker exec -w /app -e PYTHONPATH=/app langconnect-api \
        python scripts/ingest_multihop.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from langconnect.database.collections import Collection, CollectionsManager
from langconnect.database.connection import close_db_pool
from langconnect.services.document_processor import process_document

DEFAULT_COLLECTION_NAME = "multihop-rag"
DEFAULT_MANIFEST = "benchmarking/data/multihoprag/articles_manifest.json"
DEFAULT_CHUNK_SIZE = 3000
DEFAULT_CHUNK_OVERLAP = 200
EVIDENCE_METADATA_KEYS = ("source", "title", "url", "category", "published_at")


class LocalUploadFile:
    """Minimal UploadFile stand-in for a local text article."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.filename = path.name
        self.content_type = "text/plain"

    async def read(self) -> bytes:
        return self.path.read_bytes()


async def _get_or_create_collection(name: str, manifest_count: int) -> str:
    manager = CollectionsManager()
    for collection in await manager.list():
        if collection["name"] == name:
            return collection["uuid"]
    created = await manager.create(
        name,
        {
            "dataset": "MultiHop-RAG",
            "article_count": manifest_count,
            "created_for": "multihop H1 wiki-injection benchmark",
            "created_at": datetime.now().isoformat(),
        },
    )
    if not created:
        raise RuntimeError(f"Failed to create collection {name!r}")
    return created["uuid"]


def _chunk_metadata(entry: dict) -> dict:
    """Build chunk metadata carrying the evidence-match keys (source/title/url)."""
    raw = entry.get("metadata", {}) if isinstance(entry, dict) else {}
    metadata = {key: raw[key] for key in EVIDENCE_METADATA_KEYS if raw.get(key)}
    metadata["filename"] = entry.get("filename")
    metadata["source_path"] = entry.get("path")
    metadata["mime_type"] = "text/plain"
    metadata["created_at"] = datetime.now().isoformat()
    return metadata


async def ingest(args: argparse.Namespace) -> int:
    """Ingest all manifest articles into the collection."""
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        print(f"No articles in manifest {manifest_path}", file=sys.stderr)
        return 2

    collection_id = await _get_or_create_collection(args.collection_name, len(entries))
    collection = Collection(collection_id=collection_id)
    print(
        f"ingest_start collection={args.collection_name} id={collection_id} "
        f"articles={len(entries)}",
        flush=True,
    )

    processed = total_chunks = failed = 0
    for index, entry in enumerate(entries, start=1):
        path = entry.get("path")
        article_path = repo_root / path if path else None
        if not article_path or not article_path.is_file():
            failed += 1
            print(f"FAIL {path}: file not found", file=sys.stderr, flush=True)
            continue
        try:
            docs = await process_document(
                LocalUploadFile(article_path),
                metadata=_chunk_metadata(entry),
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                collection_id=collection_id,
            )
            added_ids = await collection.upsert(docs)
            processed += 1
            total_chunks += len(added_ids)
            if index % 25 == 0 or index == len(entries):
                print(
                    f"  ...{index}/{len(entries)} chunks_so_far={total_chunks}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {article_path.name}: {exc!r}", file=sys.stderr, flush=True)

    print(
        f"ingest_done collection_id={collection_id} processed={processed} "
        f"failed={failed} chunks={total_chunks}",
        flush=True,
    )
    return 0 if failed == 0 else 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    return parser.parse_args()


async def _main() -> int:
    try:
        return await ingest(parse_args())
    finally:
        await close_db_pool()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
