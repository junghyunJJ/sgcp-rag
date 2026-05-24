"""MultiHop-RAG H1 A/B: does injecting wiki summaries into generation help?

Runs the agentic pipeline over a MultiHop-RAG pilot in selectable lanes:
- off : use_wiki_context=False (no wiki)
- on  : use_wiki_context=True  (wiki promotion; + WIKI_CONTEXT_INJECT=true injects
        the selected page summaries into the generation prompt -- H1)

Because injection is env-gated in the generate node, isolate H1 with two runs:
  # promotion-only baseline (also gives the off control)
  docker exec ... python scripts/multihop_h1_ab.py --lanes off,on
  # promotion + injection (compare its on-lane to the run above)
  docker exec ... -e WIKI_CONTEXT_INJECT=true python scripts/multihop_h1_ab.py --lanes on

Scoring reuses benchmark_multihop_wiki.py: token_f1, contains, evidence_recall.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from scripts.benchmark_multihop_wiki import (
    evidence_recall,
    load_cases,
    normalize_answer,
    token_f1,
)

from langconnect.agent import run_agentic_search
from langconnect.database.connection import close_db_pool

DEFAULT_COLLECTION = "29ee1f13-2b5c-4e2b-8dff-26af9ad00ac7"
DEFAULT_DATASET = "benchmarking/data/multihoprag/MultiHopRAG.json"
DEFAULT_REPORT = "/tmp/multihop_h1_ab.jsonl"


def _contains(answer: str, gold: str) -> bool:
    na, ng = normalize_answer(answer), normalize_answer(gold)
    return bool(ng) and (ng in na or na in ng)


async def run_lane(case: Any, collection_id: str, use_wiki: bool, max_rewrites: int) -> dict:
    """Run one lane for one case and score it."""
    result = await run_agentic_search(
        question=case.question,
        collection_id=collection_id,
        use_wiki_context=use_wiki,
        max_rewrites=max_rewrites,
    )
    answer = result.get("generation") or ""
    relevant = result.get("relevant_documents", [])
    return {
        "answer": answer[:300],
        "token_f1": token_f1(answer, case.answer),
        "contains": _contains(answer, case.answer),
        "evidence_recall": evidence_recall(case.expected_document_keys, relevant),
        "wiki_status": result.get("wiki_context_status"),
        "wiki_injected": any(
            "injected into generation prompt" in s for s in result.get("steps", [])
        ),
        "error": result.get("error"),
    }


def summarize(rows: list[dict], lane: str) -> None:
    """Print mean metrics for one lane."""
    vals = [r[lane] for r in rows if lane in r]
    n = len(vals)
    errors = sum(1 for v in vals if v.get("error"))
    f1 = sum(v["token_f1"] for v in vals) / n if n else 0.0
    contains = sum(1 for v in vals if v["contains"])
    recs = [v["evidence_recall"] for v in vals if v["evidence_recall"] is not None]
    ev = sum(recs) / len(recs) if recs else 0.0
    injected = sum(1 for v in vals if v.get("wiki_injected"))
    print(
        f"{lane:4} | n={n} errors={errors} | token_f1={f1:.3f} "
        f"| contains={contains}/{n} | evidence_recall={ev:.3f} | injected={injected}"
    )


async def run(args: argparse.Namespace) -> int:
    """Run the selected lanes over the MultiHop pilot."""
    lanes = [x.strip() for x in args.lanes.split(",") if x.strip()]
    cases = load_cases(Path(args.dataset))[: args.limit]
    print(f"cases={len(cases)} lanes={lanes} inject_env={os.getenv('WIKI_CONTEXT_INJECT')}")
    rows: list[dict] = []
    for index, case in enumerate(cases, start=1):
        row: dict[str, Any] = {"id": case.id, "question_type": case.question_type}
        for lane in lanes:
            row[lane] = await run_lane(
                case, args.collection_id, lane == "on", args.max_rewrites
            )
        rows.append(row)
        if index % 10 == 0 or index == len(cases):
            print(f"  ...{index}/{len(cases)}", flush=True)

    report = Path(args.report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print("\n=== MultiHop-RAG H1 A/B ===")
    for lane in lanes:
        summarize(rows, lane)
    print(f"report: {report}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--report-path", default=DEFAULT_REPORT)
    parser.add_argument("--lanes", default="off,on")
    parser.add_argument("--max-rewrites", type=int, default=1)
    parser.add_argument("--limit", type=int, default=50)
    return parser.parse_args()


async def _main() -> int:
    try:
        return await run(parse_args())
    finally:
        await close_db_pool()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
