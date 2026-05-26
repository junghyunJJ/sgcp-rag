"""Ingest MDA-QA pilot papers (text from SPIQA paragraphs) into a collection.

MDA-QA (HF YeloDriver/MDAQA) gives questions with `support` = lists of arXiv ids.
The paper TEXT comes from SPIQA's extracted paragraphs (HF google/spiqa,
SPIQA_train_val_test-A_extracted_paragraphs.zip, one {arxiv_id}vN.txt per paper).

This ingests the union of support papers for the first --num-questions questions
(a pilot corpus with natural cross-question distractors), as full text at
chunk_size 3000 / overlap 200, attaching `arxiv_id` metadata so the benchmark's
evidence_recall can match (MDA-QA evidence keys = support arXiv ids).

Run inside the API container:
    docker exec -w /app -e PYTHONPATH=/app langconnect-api \
        python scripts/ingest_mdaqa.py --num-questions 200
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from huggingface_hub import hf_hub_download

from langconnect.database.collections import Collection, CollectionsManager
from langconnect.database.connection import close_db_pool
from langconnect.services.document_processor import process_document

DEFAULT_COLLECTION_NAME = "mdaqa-pilot"
DEFAULT_NUM_QUESTIONS = 200
DEFAULT_CHUNK_SIZE = 3000
DEFAULT_CHUNK_OVERLAP = 200
SPIQA_ZIP_GLOB = (
    "/home/langconnect/.cache/huggingface/hub/datasets--google--spiqa/"
    "snapshots/*/SPIQA_train_val_test-A_extracted_paragraphs.zip"
)


class LocalTextUpload:
    """Minimal UploadFile stand-in backed by in-memory text bytes."""

    def __init__(self, name: str, text: str) -> None:
        self.filename = name
        self.content_type = "text/plain"
        self._data = text.encode("utf-8")

    async def read(self) -> bytes:
        return self._data


def _base_arxiv(member: str) -> str:
    return re.sub(r"v\d+$", "", member.split("/")[-1][:-4])


def _build_spiqa_map() -> tuple[object, dict[str, str]]:
    import zipfile

    zip_path = glob.glob(SPIQA_ZIP_GLOB)
    if not zip_path:
        raise FileNotFoundError("SPIQA extracted_paragraphs zip not found in HF cache")
    zf = zipfile.ZipFile(zip_path[0])
    mapping: dict[str, str] = {}
    for name in zf.namelist():
        if name.endswith(".txt"):
            mapping.setdefault(_base_arxiv(name), name)
    return zf, mapping


def _pilot_papers(num_questions: int) -> list[str]:
    path = hf_hub_download("YeloDriver/MDAQA", "MDA-QA.json", repo_type="dataset")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    papers: list[str] = []
    seen: set[str] = set()
    for record in data[:num_questions]:
        for arxiv_id in record.get("support", []):
            key = str(arxiv_id)
            if key not in seen:
                seen.add(key)
                papers.append(key)
    return papers


async def _get_or_create_collection(name: str, paper_count: int) -> str:
    manager = CollectionsManager()
    for collection in await manager.list():
        if collection["name"] == name:
            return collection["uuid"]
    created = await manager.create(
        name,
        {
            "dataset": "MDA-QA",
            "paper_count": paper_count,
            "created_for": "MDA-QA academic multi-hop wiki benchmark (pilot)",
            "created_at": datetime.now().isoformat(),
        },
    )
    if not created:
        raise RuntimeError(f"Failed to create collection {name!r}")
    return created["uuid"]


async def ingest(args: argparse.Namespace) -> int:
    """Ingest the pilot support papers as full text."""
    papers = _pilot_papers(args.num_questions)
    zf, spiqa = _build_spiqa_map()
    covered = [p for p in papers if p in spiqa]
    missing = len(papers) - len(covered)
    if not covered:
        print("No covered papers to ingest.", file=sys.stderr)
        return 2

    collection_id = await _get_or_create_collection(args.collection_name, len(covered))
    collection = Collection(collection_id=collection_id)
    print(
        f"ingest_start collection={args.collection_name} id={collection_id} "
        f"questions={args.num_questions} papers={len(covered)} missing={missing} "
        f"chunk_size={args.chunk_size}",
        flush=True,
    )

    processed = total_chunks = failed = 0
    for index, arxiv_id in enumerate(covered, start=1):
        try:
            text = zf.read(spiqa[arxiv_id]).decode("utf-8", "ignore")
            docs = await process_document(
                LocalTextUpload(f"{arxiv_id}.txt", text),
                metadata={
                    "arxiv_id": arxiv_id,
                    "source": arxiv_id,
                    "dataset": "MDA-QA",
                    "mime_type": "text/plain",
                    "created_at": datetime.now().isoformat(),
                },
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                collection_id=collection_id,
            )
            added = await collection.upsert(docs)
            processed += 1
            total_chunks += len(added)
            if index % 25 == 0 or index == len(covered):
                print(f"  ...{index}/{len(covered)} chunks_so_far={total_chunks}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {arxiv_id}: {exc!r}", file=sys.stderr, flush=True)

    print(
        f"ingest_done collection_id={collection_id} papers={processed} "
        f"failed={failed} total_chunks={total_chunks} "
        f"chunks_per_paper={total_chunks / processed:.1f}"
        if processed
        else "ingest_done (no papers)",
        flush=True,
    )
    return 0 if failed == 0 else 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--num-questions", type=int, default=DEFAULT_NUM_QUESTIONS)
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
