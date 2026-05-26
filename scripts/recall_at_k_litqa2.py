"""Retrieval recall@k for LitQA2: wiki OFF vs ON (source-ref promotion).

Measures whether each question's gold `key-passage` appears in the retrieved
top-k chunks, comparing plain hybrid search (wiki OFF) against search plus LLM
Wiki source-ref promotion (wiki ON). Retrieval only -- no generation, no judge.

Run inside the API container (local PubMedBERT embeddings, no external key):

    docker cp scripts/recall_at_k_litqa2.py langconnect-api:/app/scripts/
    docker exec -w /app langconnect-api python scripts/recall_at_k_litqa2.py \
        --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from langconnect.agent import _resolve_wiki_promotion
from langconnect.agent.wiki_context import resolve_wiki_context
from langconnect.database.collections import Collection
from langconnect.database.connection import close_db_pool

DEFAULT_COLLECTION = "b090b794-07fb-4194-9bfa-8914d98b864e"
DEFAULT_JSONL = "benchmarking/data/lab-bench/LitQA2/litqa2-public.jsonl"
DEFAULT_WIKI_DIR = "llm_wiki/collections"
DEFAULT_REPORT = "reports/recall_at_k_litqa2.jsonl"
DEFAULT_KS = (5, 10)


def _normalize(text: str) -> str:
    """Lowercase and collapse non-alphanumerics (PDF hyphenation/whitespace)."""
    text = re.sub(r"[^a-z0-9]+", " ", text.casefold())
    return re.sub(r"\s+", " ", text).strip()


TOKEN_OVERLAP_THRESHOLD = 0.6


def passage_hit(key_passage: str, chunk_text: str) -> bool:
    """Return True if `chunk_text` contains the gold `key_passage`.

    Uses token overlap rather than verbatim substring, since PDF->markdown->chunk
    extraction alters whitespace and hyphenation. A hit means the fraction of
    distinct normalized key-passage tokens also present in the chunk is at least
    TOKEN_OVERLAP_THRESHOLD (0.6).
    """
    key_tokens = set(_normalize(key_passage).split())
    if not key_tokens:
        return False
    chunk_tokens = set(_normalize(chunk_text).split())
    overlap = len(key_tokens & chunk_tokens) / len(key_tokens)
    return overlap >= TOKEN_OVERLAP_THRESHOLD


def load_questions(path: Path, limit: int | None) -> list[dict[str, str]]:
    """Load LitQA2 records that carry a gold key-passage."""
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key_passage = (record.get("key-passage") or "").strip()
            if not key_passage:
                continue
            rows.append(
                {
                    "id": record["id"],
                    "question": record["question"],
                    "key_passage": key_passage,
                }
            )
            if limit and len(rows) >= limit:
                break
    return rows


async def eval_one(
    collection: Collection,
    collection_id: str,
    question: str,
    max_k: int,
    wiki_dir: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Run wiki-OFF search and wiki-ON promotion for one question."""
    search_docs = await collection.search(
        question, limit=max_k, search_type="hybrid"
    )
    wiki_result = resolve_wiki_context(collection_id, question, wiki_dir=wiki_dir)
    _, promoted, wiki_status, _ = await _resolve_wiki_promotion(
        collection_id, wiki_result, question
    )
    return search_docs, promoted, wiki_status


def score_one(
    record: dict[str, str],
    search_docs: list[dict[str, Any]],
    promoted: list[dict[str, Any]],
    wiki_status: str,
    ks: tuple[int, ...],
) -> dict[str, Any]:
    """Compute per-question off/on/recovered flags for each k."""
    key_passage = record["key_passage"]
    promoted_hit = any(
        passage_hit(key_passage, doc.get("page_content", "")) for doc in promoted
    )
    result: dict[str, Any] = {
        "id": record["id"],
        "wiki_status": wiki_status,
        "n_promoted": len(promoted),
        "promoted_hit": promoted_hit,
    }
    for k in ks:
        off = any(
            passage_hit(key_passage, doc.get("page_content", ""))
            for doc in search_docs[:k]
        )
        result[f"off@{k}"] = off
        result[f"on@{k}"] = off or promoted_hit
        result[f"recovered@{k}"] = (not off) and promoted_hit
    return result


def print_summary(rows: list[dict[str, Any]], ks: tuple[int, ...]) -> None:
    """Print aggregate recall@k off vs on and wiki-promotion contribution."""
    n = len(rows)
    print(f"\n=== recall@k over {n} LitQA2 questions ===")
    for k in ks:
        off = sum(r[f"off@{k}"] for r in rows)
        on = sum(r[f"on@{k}"] for r in rows)
        recovered = sum(r[f"recovered@{k}"] for r in rows)
        print(
            f"k={k:>2} | wiki OFF recall {off / n:.3f} ({off}/{n}) "
            f"| wiki ON recall {on / n:.3f} ({on}/{n}) "
            f"| promotion recovered {recovered}"
        )
    print("wiki status:", dict(Counter(r["wiki_status"] for r in rows)))
    print("questions with promoted gold chunk:", sum(r["promoted_hit"] for r in rows))


async def run(args: argparse.Namespace) -> int:
    """Evaluate retrieval recall@k for the configured LitQA2 collection."""
    ks = tuple(int(k) for k in args.ks.split(","))
    max_k = max(ks)
    records = load_questions(Path(args.jsonl), args.limit)
    if not records:
        print("No LitQA2 records with key-passage found.")
        return 1

    collection = Collection(collection_id=args.collection_id)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        search_docs, promoted, wiki_status = await eval_one(
            collection, args.collection_id, record["question"], max_k, args.wiki_dir
        )
        rows.append(score_one(record, search_docs, promoted, wiki_status, ks))
        if index % 10 == 0:
            print(f"  ...{index}/{len(records)}")

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print_summary(rows, ks)
    print(f"\nPer-question report written to {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION)
    parser.add_argument("--jsonl", default=DEFAULT_JSONL)
    parser.add_argument("--wiki-dir", default=DEFAULT_WIKI_DIR)
    parser.add_argument("--report-path", default=DEFAULT_REPORT)
    parser.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


async def _main() -> int:
    try:
        return await run(parse_args())
    finally:
        await close_db_pool()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
