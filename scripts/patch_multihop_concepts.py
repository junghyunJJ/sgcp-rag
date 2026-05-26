"""Patch concept pages into an existing LLM Wiki runtime pack.

The MultiHop wiki was built with 0 concept pages (concept-ref bug, since fixed).
Rather than re-running the full 609-source rebuild, regenerate ONLY the concept
pages from the source pages already in the pack and append them to the runtime
pack ({collection_id}.json) that resolve_wiki_context reads. Source pages are
left untouched. (The manifest/markdown artifacts are not updated -- the agentic
runtime only consumes the runtime pack.)

    docker cp scripts/patch_multihop_concepts.py langconnect-api:/app/scripts/
    docker exec -w /app -e PYTHONPATH=/app langconnect-api \
        python scripts/patch_multihop_concepts.py --collection-id <id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

from langconnect.agent.config import get_agent_llm
from langconnect.agent.wiki_context import _validate_pages
from langconnect.database.connection import close_db_pool
from langconnect.services import llm_wiki as W

DEFAULT_COLLECTION = "29ee1f13-2b5c-4e2b-8dff-26af9ad00ac7"
WIKI_DIR = "/app/llm_wiki/collections"


async def run(args: argparse.Namespace) -> int:
    """Regenerate concept pages and append them to the runtime pack."""
    pack_path = Path(WIKI_DIR) / f"{args.collection_id}.json"
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    collection_id = pack["collection_id"]

    source_records = [p for p in pack["pages"] if p.get("type") == "source"]
    source_pages = [
        SimpleNamespace(
            id=p["id"],
            title=p["title"],
            summary=p["summary"],
            keywords=p.get("keywords", []),
            source_refs=p.get("source_refs", []),
        )
        for p in source_records
    ]
    print(f"source pages in pack: {len(source_pages)}")

    concepts = await W._generate_concept_pages(get_agent_llm(), source_pages)
    concept_records = [W._page_pack_record(c) for c in concepts]
    print(f"generated concept pages: {len(concept_records)}")
    for c in concepts:
        print(f"  - {c.title[:50]!r} refs={len(c.source_refs)}")

    pack["pages"] = source_records + concept_records
    pack["concept_count"] = len(concept_records)
    pack["page_count"] = len(pack["pages"])

    if _validate_pages(pack, collection_id) is None:
        print("ERROR: patched pack failed schema validation; not writing.")
        return 1

    tmp = pack_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, pack_path)
    print(f"patched pack written: {len(pack['pages'])} pages -> {pack_path}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION)
    return parser.parse_args()


async def _main() -> int:
    try:
        return await run(parse_args())
    finally:
        await close_db_pool()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
