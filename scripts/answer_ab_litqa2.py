"""Answer-quality A/B for LitQA2: wiki OFF vs ON, scored by LLM-judge MC accuracy.

Runs the full agentic RAG pipeline twice per question (use_wiki_context False vs
True), then an LLM judge maps each free-form answer onto the LitQA2 multiple-choice
options (ideal + distractors + an "insufficient information" option). The wiki ON
lane uses the shipped default wiki (lexical token-overlap selection + frozen
source-ref promotion) -- the configuration most favorable to the wiki by recall.

Run inside the API container (uses the env-configured agent LLM for both
generation and judging):

    docker cp scripts/answer_ab_litqa2.py langconnect-api:/app/scripts/
    docker exec -w /app -e PYTHONPATH=/app langconnect-api \
        python scripts/answer_ab_litqa2.py --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any

from langconnect.agent import run_agentic_search
from langconnect.agent.config import get_agent_llm
from langconnect.database.connection import close_db_pool

DEFAULT_COLLECTION = "b090b794-07fb-4194-9bfa-8914d98b864e"
DEFAULT_JSONL = "benchmarking/data/lab-bench/LitQA2/litqa2-public.jsonl"
DEFAULT_REPORT = "/tmp/answer_ab_litqa2.jsonl"
INSUFFICIENT = "Insufficient information to answer this question"

JUDGE_PROMPT = """You are scoring a candidate answer against multiple-choice options.

Question: {question}

Candidate answer: {answer}

Options:
{options}

Reply with ONLY the single capital letter of the option that best matches the \
candidate answer. If the candidate gives no clear or supported answer, reply with \
the letter of the "Insufficient information" option."""


def load_questions(path: Path, limit: int | None) -> list[dict[str, Any]]:
    """Load LitQA2 records carrying an ideal answer and distractors."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ideal = (record.get("ideal") or "").strip()
            distractors = [
                str(d).strip() for d in record.get("distractors", []) if str(d).strip()
            ]
            if not ideal or not distractors:
                continue
            rows.append(
                {
                    "id": record["id"],
                    "question": record["question"],
                    "ideal": ideal,
                    "distractors": distractors,
                }
            )
            if limit and len(rows) >= limit:
                break
    return rows


def build_options(
    record: dict[str, Any],
    rng: random.Random,
) -> tuple[list[tuple[str, str]], str]:
    """Return shuffled (letter, text) options plus the ideal answer's letter."""
    texts = [record["ideal"], *record["distractors"], INSUFFICIENT]
    rng.shuffle(texts)
    options = [(chr(65 + i), text) for i, text in enumerate(texts)]
    ideal_letter = next(letter for letter, text in options if text == record["ideal"])
    return options, ideal_letter


async def judge_answer(
    llm: Any,
    question: str,
    answer: str,
    options: list[tuple[str, str]],
) -> str | None:
    """Ask the LLM judge which option a free-form answer corresponds to."""
    block = "\n".join(f"{letter}. {text}" for letter, text in options)
    prompt = JUDGE_PROMPT.format(
        question=question,
        answer=answer or "(no answer given)",
        options=block,
    )
    response = await llm.ainvoke(prompt)
    content = getattr(response, "content", response)
    match = re.search(r"[A-Z]", str(content).upper())
    return match.group(0) if match else None


async def run_question(
    record: dict[str, Any],
    collection_id: str,
    max_rewrites: int,
    llm: Any,
    rng: random.Random,
) -> dict[str, Any]:
    """Run both wiki lanes for one question and judge each answer."""
    options, ideal_letter = build_options(record, rng)
    out: dict[str, Any] = {"id": record["id"], "ideal_letter": ideal_letter}
    for lane, use_wiki in (("off", False), ("on", True)):
        result = await run_agentic_search(
            question=record["question"],
            collection_id=collection_id,
            use_wiki_context=use_wiki,
            max_rewrites=max_rewrites,
        )
        generation = result.get("generation") or ""
        picked = await judge_answer(llm, record["question"], generation, options)
        out[f"{lane}_picked"] = picked
        out[f"{lane}_correct"] = picked == ideal_letter
        out[f"{lane}_no_context"] = bool(result.get("no_context_found"))
        out[f"{lane}_wiki_status"] = result.get("wiki_context_status")
        out[f"{lane}_gen"] = generation[:300]
    return out


def print_summary(rows: list[dict[str, Any]]) -> None:
    """Print OFF vs ON accuracy and the answer-flip breakdown."""
    n = len(rows)
    off = sum(r["off_correct"] for r in rows)
    on = sum(r["on_correct"] for r in rows)
    off_to_on = sum(1 for r in rows if not r["off_correct"] and r["on_correct"])
    on_to_off = sum(1 for r in rows if r["off_correct"] and not r["on_correct"])
    print(f"\n=== answer-quality A/B over {n} LitQA2 questions ===")
    print(f"wiki OFF accuracy: {off / n:.3f} ({off}/{n})")
    print(f"wiki ON  accuracy: {on / n:.3f} ({on}/{n})")
    print(f"delta: {(on - off) / n:+.3f}  (OFF->ON fixed {off_to_on}, broke {on_to_off})")


async def run(args: argparse.Namespace) -> int:
    """Run the LitQA2 answer-quality A/B and write a per-question report."""
    records = load_questions(Path(args.jsonl), args.limit)
    if not records:
        print("No LitQA2 records with ideal+distractors found.")
        return 1

    llm = get_agent_llm()
    rng = random.Random(0)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        rows.append(
            await run_question(record, args.collection_id, args.max_rewrites, llm, rng)
        )
        print(f"  ...{index}/{len(records)}", flush=True)

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print_summary(rows)
    print(f"\nPer-question report written to {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION)
    parser.add_argument("--jsonl", default=DEFAULT_JSONL)
    parser.add_argument("--report-path", default=DEFAULT_REPORT)
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
