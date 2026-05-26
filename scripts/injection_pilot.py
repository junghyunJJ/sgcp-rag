"""Controlled injection-variant PILOT for a paper-grade claim (exploratory).

Noise-free design: one agentic run per question fixes the graded context, then
several generations are produced on that SAME context, varying ONLY the injected
block. A->A2 (both no-injection) measures single-call noise; A->variant measures
each injection variant. Run with WIKI_SEMANTIC_SELECT=true so selection scores are
cosines usable for gating.

Variants per question (same fixed context C):
  A       no injection (baseline)
  A2      no injection, re-call (NOISE baseline)
  B_all   inject all selected summaries, default framing (= production)
  B_frame inject all selected summaries, STRONG anti-override framing
  B_gated inject only the top-1 (highest-score) selected summary

Outputs per-variant fixed/broke vs A, McNemar exact p, and a confirmatory-N power
estimate from the best variant. This pilot selects a variant + sizes the confirm
run; it makes NO claim itself.

    docker cp scripts/injection_pilot.py langconnect-api:/app/scripts/
    docker exec -w /app -e PYTHONPATH=/app -e WIKI_SEMANTIC_SELECT=true \
        langconnect-api python scripts/injection_pilot.py --limit 80 \
        --report-path /tmp/injection_pilot.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
from math import comb, sqrt
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from scripts.mdaqa_h1_ab import judge_correct, load_cases

from langconnect.agent import run_agentic_search
from langconnect.agent.config import get_agent_llm
from langconnect.agent.prompts import ANSWER_GENERATOR_PROMPT
from langconnect.agent.wiki_context import _render_context, resolve_wiki_context
from langconnect.database.connection import close_db_pool

DEFAULT_COLLECTION = "28d0e6e0-99f1-4b03-b7ed-ebfdcd7371f1"

DEFAULT_HEADER = (
    "Background orientation from a non-authoritative LLM Wiki "
    "(navigation memory only -- do NOT cite it as a source; ground your "
    "answer in the retrieved context below):"
)
STRONG_HEADER = DEFAULT_HEADER + (
    " If anything here conflicts with or is not supported by the retrieved "
    "context, ignore it and rely solely on the retrieved context; do not "
    "introduce facts that appear only here."
)
VARIANTS = ("B_all", "B_frame", "B_gated")


def _inject(header: str, pages: list[dict[str, Any]], context: str) -> str:
    return f"{header}\n{_render_context(pages)}\n\n---\n\nRetrieved context:\n{context}"


async def _generate(llm: Any, question: str, context: str) -> str:
    prompt = ChatPromptTemplate.from_messages([("human", ANSWER_GENERATOR_PROMPT)])
    result = await (prompt | llm).ainvoke({"question": question, "context": context})
    return result.content


def _mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar (sign test on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(2 * tail, 1.0)


def _required_discordant(p_fav: float) -> float | None:
    """Discordant pairs needed for power 0.8 at alpha 0.05 (two-sided), given the
    probability p_fav that a discordant pair favors the variant. None if p_fav<=0.5."""
    if p_fav <= 0.5:
        return None
    z_a, z_b = 1.959964, 0.841621
    num = z_a * 0.5 + z_b * sqrt(p_fav * (1 - p_fav))
    return (num * num) / ((p_fav - 0.5) ** 2)


def _variant_context(variant: str, pages: list[dict[str, Any]], context: str) -> str:
    if variant == "B_all":
        return _inject(DEFAULT_HEADER, pages, context)
    if variant == "B_frame":
        return _inject(STRONG_HEADER, pages, context)
    if variant == "B_gated":
        return _inject(DEFAULT_HEADER, pages[:1], context)
    raise ValueError(f"unknown variant {variant!r}")


async def run(args: argparse.Namespace) -> int:
    """Run the controlled injection-variant pilot."""
    cases = load_cases(args.start + args.limit)[args.start :]
    active = [v for v in VARIANTS if v in {x.strip() for x in args.variants.split(",")}]
    print(f"cases={len(cases)} start={args.start} variants={active}", flush=True)
    llm = get_agent_llm()
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        result = await run_agentic_search(
            question=case["question"],
            collection_id=args.collection_id,
            use_wiki_context=True,
            max_rewrites=args.max_rewrites,
        )
        relevant = result.get("relevant_documents", [])
        wiki = resolve_wiki_context(args.collection_id, case["question"])
        if not relevant or wiki.status != "selected" or not wiki.selected_pages:
            continue

        context = "\n\n---\n\n".join(d.get("page_content", "") for d in relevant)
        pages = wiki.selected_pages  # sorted by descending score
        prompts = {"A": context, "A2": context}
        for v in active:
            prompts[v] = _variant_context(v, pages, context)
        verdicts: dict[str, bool] = {}
        for key, ctx in prompts.items():
            ans = await _generate(llm, case["question"], ctx)
            verdicts[key] = await judge_correct(llm, case["question"], case["answer"], ans)

        rows.append(
            {
                "id": case["id"],
                **verdicts,
                "n_selected": len(pages),
                "selected_scores": [round(float(p.get("score", 0)), 3) for p in pages],
            }
        )
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

    def flips(variant: str) -> tuple[int, int]:
        fixed = sum(1 for r in rows if not r["A"] and r[variant])
        broke = sum(1 for r in rows if r["A"] and not r[variant])
        return fixed, broke

    nf, nb = flips("A2")
    print(f"\n=== Injection pilot (n={n} firing questions) ===")
    print(f"correct A={sum(r['A'] for r in rows)}/{n}")
    print(f"NOISE baseline A->A2: fixed={nf} broke={nb} (must be ~0 for clean isolation)")
    best = None
    for v in active:
        f, b = flips(v)
        disc = f + b
        p = _mcnemar_p(f, b)
        correct = sum(r[v] for r in rows)
        print(
            f"{v:8} correct={correct}/{n} | A->{v}: fixed={f} broke={b} "
            f"(discordant={disc}) McNemar p={p:.3f}"
        )
        if disc and (best is None or (f - b) > best[1]):
            best = (v, f - b, f, b)

    if best:
        v, net, f, b = best
        disc = f + b
        p_fav = f / disc if disc else 0.0
        firing_rate = n / args.limit
        need = _required_discordant(p_fav)
        print(f"\nbest variant: {v} (net {net:+d}, fixed {f} / broke {b}, p_fav={p_fav:.2f})")
        if need is None:
            print("  -> p_fav<=0.5: not powerable (no positive effect to confirm).")
        else:
            disc_rate = disc / n
            firing_needed = need / disc_rate if disc_rate else float("inf")
            total_needed = firing_needed / firing_rate if firing_rate else float("inf")
            print(f"  -> for power 0.8 @ alpha .05: ~{need:.0f} discordant pairs")
            print(f"     ~{firing_needed:.0f} firing questions ~= {total_needed:.0f} total "
                  f"MDA-QA questions (firing rate {firing_rate:.0%}, discordant rate "
                  f"{disc_rate:.0%})")
    print(f"report: {report}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION)
    parser.add_argument("--report-path", default="/tmp/injection_pilot.jsonl")
    parser.add_argument("--max-rewrites", type=int, default=1)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--start", type=int, default=0, help="skip first N cases (fresh slice)")
    parser.add_argument("--variants", default="B_all,B_frame,B_gated")
    return parser.parse_args()


async def _main() -> int:
    try:
        return await run(parse_args())
    finally:
        await close_db_pool()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
