"""Fetch real abstracts from arXiv for MDA-QA pilot papers.

Writes {base_arxiv_id: abstract} JSON, used as WIKI_ABSTRACT_SOURCE_FILE for the
default abstract-based wiki build (SPIQA's extracted paragraphs have no clean abstract).

    docker exec -w /app -e PYTHONPATH=/app langconnect-api \
        python scripts/fetch_arxiv_abstracts.py --num-questions 200 \
        --out /tmp/mdaqa_abstracts.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from huggingface_hub import hf_hub_download

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
BATCH = 100
SLEEP_SECONDS = 3.0


def _pilot_arxiv_ids(num_questions: int) -> list[str]:
    path = hf_hub_download("YeloDriver/MDAQA", "MDA-QA.json", repo_type="dataset")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ids: list[str] = []
    seen: set[str] = set()
    for record in data[:num_questions]:
        for arxiv_id in record.get("support", []):
            key = str(arxiv_id)
            if key not in seen:
                seen.add(key)
                ids.append(key)
    return ids


def _base_id(entry_id: str) -> str:
    tail = entry_id.rstrip("/").split("/")[-1]
    return re.sub(r"v\d+$", "", tail)


def _fetch_batch(ids: list[str], *, retries: int = 4) -> dict[str, str]:
    query = urllib.parse.urlencode(
        {"id_list": ",".join(ids), "max_results": len(ids)}
    )
    url = f"{ARXIV_API}?{query}"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mdaqa-bench/0.1"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                root = ET.fromstring(resp.read())
            break
        except Exception as exc:  # noqa: BLE001 -- arXiv 503s are transient
            last_error = exc
            time.sleep(5 * (attempt + 1))
    else:
        raise last_error or RuntimeError("arXiv fetch failed")
    out: dict[str, str] = {}
    for entry in root.findall(f"{ATOM}entry"):
        eid = entry.findtext(f"{ATOM}id") or ""
        summary = entry.findtext(f"{ATOM}summary") or ""
        summary = re.sub(r"\s+", " ", summary).strip()
        if eid and summary:
            out[_base_id(eid)] = summary
    return out


def main() -> int:
    """Fetch abstracts in batches and write the override JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--out", default="/tmp/mdaqa_abstracts.json")
    args = parser.parse_args()

    ids = _pilot_arxiv_ids(args.num_questions)
    print(f"fetching abstracts for {len(ids)} papers...", flush=True)
    abstracts: dict[str, str] = {}
    for start in range(0, len(ids), BATCH):
        batch = ids[start : start + BATCH]
        try:
            abstracts.update(_fetch_batch(batch))
        except Exception as exc:  # noqa: BLE001
            print(f"batch {start} failed: {exc!r}", file=sys.stderr, flush=True)
        print(f"  ...{min(start + BATCH, len(ids))}/{len(ids)} got {len(abstracts)}", flush=True)
        if start + BATCH < len(ids):
            time.sleep(SLEEP_SECONDS)

    Path(args.out).write_text(json.dumps(abstracts, ensure_ascii=False), encoding="utf-8")
    missing = [i for i in ids if i not in abstracts]
    print(
        f"done: {len(abstracts)}/{len(ids)} abstracts written to {args.out}; "
        f"missing={len(missing)} {missing[:5]}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
