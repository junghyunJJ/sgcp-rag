"""Controlled test of wiki summary injection (H1), isolated from pipeline noise.

The full agentic loop is nondeterministic (two identical runs differ on ~74% of
answers), which swamps any injection effect. Injection only changes ONE step --
the generation prompt -- so this test freezes retrieval/grading (one agentic run
per question) and varies ONLY the generation prompt on that fixed context:

  A   = generate(question, context)              # injection OFF
  A'  = generate(question, context)              # injection OFF, re-call -> noise baseline
  B   = generate(question, wiki_summary + context)  # injection ON

A->A' flips measure single-call nondeterminism; A->B flips measure injection.
If A->B exceeds the A->A' baseline (and is asymmetric), injection helps; if not,
it is within noise. Only questions where injection would actually fire (wiki page
selected, non-empty graded context) are scored.

    docker cp scripts/controlled_injection_test.py langconnect-api:/app/scripts/
    docker exec -w /app -e PYTHONPATH=/app langconnect-api \
        python scripts/controlled_injection_test.py --limit 50 --report-path /tmp/ci.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from scripts.mdaqa_h1_ab import judge_correct, load_cases

from langconnect.agent import run_agentic_search
from langconnect.agent.config import get_agent_llm
from langconnect.agent.prompts import ANSWER_GENERATOR_PROMPT
from langconnect.agent.wiki_context import resolve_wiki_context
from langconnect.database.connection import close_db_pool

DEFAULT_COLLECTION = "28d0e6e0-99f1-4b03-b7ed-ebfdcd7371f1"

# Must mirror the injection block in langconnect/agent/nodes.py generate().
INJECT_TEMPLATE = (
    "Background orientation from a non-authoritative LLM Wiki "
    "(navigation memory only -- do NOT cite it as a source; ground your "
    "answer in the retrieved context below):\n{wiki}\n\n---\n\n"
    "Retrieved context:\n{context}"
)


async def _generate(llm: Any, question: str, context: str) -> str:
    prompt = ChatPromptTemplate.from_messages([("human", ANSWER_GENERATOR_PROMPT)])
    result = await (prompt | llm).ainvoke({"question": question, "context": context})
    return result.content


async def run(args: argparse.Namespace) -> int:
    """Score injection vs a same-input noise baseline on fixed contexts."""
    cases = load_cases(args.limit)
    llm = get_agent_llm()
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        # 1) Fixed context: one agentic run (wiki on -> promotion applied).
        result = await run_agentic_search(
            question=case["question"],
            collection_id=args.collection_id,
            use_wiki_context=True,
            max_rewrites=args.max_rewrites,
        )
        relevant = result.get("relevant_documents", [])
        wiki = resolve_wiki_context(args.collection_id, case["question"])
        # Only score where injection would actually fire.
        if not relevant or wiki.status != "selected" or not wiki.context:
            continue

        context = "\n\n---\n\n".join(d.get("page_content", "") for d in relevant)
        context_inject = INJECT_TEMPLATE.format(wiki=wiki.context, context=context)

        # 2) Three controlled generations on the SAME fixed context.
        ans_a = await _generate(llm, case["question"], context)
        ans_a2 = await _generate(llm, case["question"], context)
        ans_b = await _generate(llm, case["question"], context_inject)

        # 3) Judge each against gold.
        gold = case["answer"]
        row = {
            "id": case["id"],
            "A": await judge_correct(llm, case["question"], gold, ans_a),
            "A2": await judge_correct(llm, case["question"], gold, ans_a2),
            "B": await judge_correct(llm, case["question"], gold, ans_b),
            "n_selected": len(wiki.selected_pages),
            "ans_a": ans_a[:200],
            "ans_b": ans_b[:200],
        }
        rows.append(row)
        if index % 10 == 0 or index == len(cases):
            print(f"  ...{index}/{len(cases)} (scored {len(rows)})", flush=True)

    report = Path(args.report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    n = len(rows)
    if n == 0:
        print("No injection-firing questions scored.")
        return 1
    noise_fixed = sum(1 for r in rows if not r["A"] and r["A2"])
    noise_broke = sum(1 for r in rows if r["A"] and not r["A2"])
    inj_fixed = sum(1 for r in rows if not r["A"] and r["B"])
    inj_broke = sum(1 for r in rows if r["A"] and not r["B"])
    print(f"\n=== Controlled injection test (n={n} injection-firing questions) ===")
    print(f"correct: A={sum(r['A'] for r in rows)}  A'={sum(r['A2'] for r in rows)}  "
          f"B={sum(r['B'] for r in rows)}  / {n}")
    print(f"NOISE baseline  A->A' : fixed={noise_fixed} broke={noise_broke} "
          f"(total flips {noise_fixed + noise_broke})")
    print(f"INJECTION       A->B  : fixed={inj_fixed} broke={inj_broke} "
          f"(total flips {inj_fixed + inj_broke})")
    print(f"report: {report}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION)
    parser.add_argument("--report-path", default="/tmp/controlled_injection.jsonl")
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
