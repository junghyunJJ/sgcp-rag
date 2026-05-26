"""MDA-QA academic multi-hop A/B: does the wiki help, and full-text vs abstract?

Lanes are selected by --lanes (off/on). The wiki variant and promotion mode are
controlled by ENV at invocation, so the full matrix is run as multiple processes:
  - wiki variant:  LANGCONNECT_WIKI_CONTEXT_DIR=llm_wiki/collections (full-text)
                   LANGCONNECT_WIKI_CONTEXT_DIR=/tmp/wiki_abstract/collections (abstract)
  - promotion:     WIKI_DOC_ROUTING=true  (re-search body; needed for long papers)
  - injection:     WIKI_CONTEXT_INJECT=true  (H1 summary injection)

Answers are ~paragraph-length, so correctness is judged by the agent LLM
(consistency with the gold answer); token_f1 and evidence_recall (arxiv_id) are
recorded as secondary signals.

    docker exec -w /app -e PYTHONPATH=/app [-e <lane envs>] langconnect-api \
        python scripts/mdaqa_h1_ab.py --lanes off,on --limit 50 --report-path /tmp/x.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from scripts.benchmark_multihop_wiki import normalize_answer, token_f1

from langconnect.agent import run_agentic_search
from langconnect.agent.config import get_agent_llm
from langconnect.database.connection import close_db_pool

DEFAULT_COLLECTION = "28d0e6e0-99f1-4b03-b7ed-ebfdcd7371f1"
DEFAULT_REPORT = "/tmp/mdaqa_h1_ab.jsonl"

JUDGE_PROMPT = """You are grading a candidate answer to a research question.

Question: {question}

Reference answer: {gold}

Candidate answer: {candidate}

Does the candidate answer correctly and sufficiently answer the question, \
consistent with the reference answer? Reply with ONLY 'YES' or 'NO'."""


def load_cases(num: int) -> list[dict[str, Any]]:
    """Load the first `num` MDA-QA questions with support arxiv ids."""
    path = hf_hub_download("YeloDriver/MDAQA", "MDA-QA.json", repo_type="dataset")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = []
    for record in data[:num]:
        cases.append(
            {
                "id": str(record.get("id")),
                "question": record["question"],
                "answer": record["answer"],
                "support": [str(s) for s in record.get("support", [])],
            }
        )
    return cases


def evidence_recall(support: list[str], relevant_docs: list[dict[str, Any]]) -> float | None:
    """Fraction of support arxiv ids present in the relevant documents."""
    expected = set(support)
    if not expected:
        return None
    observed: set[str] = set()
    for doc in relevant_docs:
        meta = doc.get("metadata") or {}
        arxiv_id = meta.get("arxiv_id")
        if arxiv_id:
            observed.add(str(arxiv_id))
    return len(expected & observed) / len(expected)


async def judge_correct(llm: Any, question: str, gold: str, candidate: str) -> bool:
    """LLM judge: is the candidate answer consistent with the gold answer?"""
    if not candidate.strip():
        return False
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, candidate=candidate)
    response = await llm.ainvoke(prompt)
    content = str(getattr(response, "content", response)).upper()
    return content.lstrip().startswith("YES") or "YES" in content[:8]


async def run_lane(case: dict[str, Any], cid: str, use_wiki: bool, max_rewrites: int, llm: Any) -> dict[str, Any]:
    """Run one lane for one case and score it."""
    result = await run_agentic_search(
        question=case["question"],
        collection_id=cid,
        use_wiki_context=use_wiki,
        max_rewrites=max_rewrites,
    )
    answer = result.get("generation") or ""
    relevant = result.get("relevant_documents", [])
    return {
        "answer": answer[:300],
        "correct": await judge_correct(llm, case["question"], case["answer"], answer),
        "token_f1": token_f1(answer, case["answer"]),
        "evidence_recall": evidence_recall(case["support"], relevant),
        "wiki_status": result.get("wiki_context_status"),
        "injected": any("injected into generation prompt" in s for s in result.get("steps", [])),
        "no_context": bool(result.get("no_context_found")),
    }


def summarize(rows: list[dict[str, Any]], lane: str) -> None:
    """Print mean metrics for one lane."""
    vals = [r[lane] for r in rows if lane in r]
    n = len(vals) or 1
    correct = sum(1 for v in vals if v["correct"])
    answered = sum(1 for v in vals if not v["no_context"])
    f1 = sum(v["token_f1"] for v in vals) / n
    recs = [v["evidence_recall"] for v in vals if v["evidence_recall"] is not None]
    ev = sum(recs) / len(recs) if recs else 0.0
    injected = sum(1 for v in vals if v["injected"])
    print(
        f"{lane:4} | n={len(vals)} answered={answered} | correct(judge)={correct} "
        f"| token_f1={f1:.3f} | evidence_recall={ev:.3f} | injected={injected}"
    )


async def run(args: argparse.Namespace) -> int:
    """Run the selected lanes over the MDA-QA pilot."""
    lanes = [x.strip() for x in args.lanes.split(",") if x.strip()]
    cases = load_cases(args.limit)
    print(
        f"cases={len(cases)} lanes={lanes} "
        f"wiki_dir={os.getenv('LANGCONNECT_WIKI_CONTEXT_DIR', 'default')} "
        f"inject={os.getenv('WIKI_CONTEXT_INJECT')} routing={os.getenv('WIKI_DOC_ROUTING')}",
        flush=True,
    )
    llm = get_agent_llm()
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        row: dict[str, Any] = {"id": case["id"]}
        for lane in lanes:
            row[lane] = await run_lane(case, args.collection_id, lane == "on", args.max_rewrites, llm)
        rows.append(row)
        if index % 10 == 0 or index == len(cases):
            print(f"  ...{index}/{len(cases)}", flush=True)

    report = Path(args.report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print("\n=== MDA-QA A/B ===")
    for lane in lanes:
        summarize(rows, lane)
    print(f"report: {report}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION)
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
