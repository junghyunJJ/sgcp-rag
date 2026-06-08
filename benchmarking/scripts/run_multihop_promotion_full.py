"""Resumable MultiHop-RAG promotion benchmark runner.

Default full run from the API container:

    docker exec -i \
      -e PYTHONPATH=/app \
      -w /app \
      langconnect-api \
      .venv/bin/python benchmarking/scripts/run_multihop_promotion_full.py

Useful targeted runs:

    # smoke test without touching the default full output directory
    .venv/bin/python benchmarking/scripts/run_multihop_promotion_full.py \
      --limit 3 --results-dir /tmp/multihop_promotion_smoke

    # one model / one condition; rerun the same command to resume
    .venv/bin/python benchmarking/scripts/run_multihop_promotion_full.py \
      --models qwen35_122b_port6000 --conditions wiki_doc_routing
"""

# ruff: noqa: E402, I001, INP001

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
with suppress(ValueError):
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_multihop_wiki import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SEARCH_TYPE,
    BenchmarkCase,
    evidence_recall,
    load_cases,
    normalize_answer,
    token_f1,
)

from langconnect.agent import run_agentic_search
from langconnect.database.connection import close_db_pool

DEFAULT_COLLECTION = "29ee1f13-2b5c-4e2b-8dff-26af9ad00ac7"
DEFAULT_DATASET = "benchmarking/data/multihoprag/MultiHopRAG.json"
DEFAULT_RESULTS_DIR = "benchmarking/results/promotion"
DEFAULT_MAX_REWRITES = 3
DEFAULT_RETRY_LIMIT = 2
DEFAULT_CASE_TIMEOUT_SECONDS = 1800.0
NO_CONTEXT_ERROR = "no_relevant_context"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    base_url: str
    model: str


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    use_wiki_context: bool
    wiki_doc_routing: bool


MODELS: dict[str, ModelSpec] = {
    "qwen35_122b_port6000": ModelSpec(
        key="qwen35_122b_port6000",
        base_url="http://host.docker.internal:6000",
        model="qwen3.5:122b",
    ),
    "qwen35_35b_port7000": ModelSpec(
        key="qwen35_35b_port7000",
        base_url="http://host.docker.internal:7000",
        model="qwen3.5:35b",
    ),
}

CONDITIONS: dict[str, ConditionSpec] = {
    "wiki_off": ConditionSpec(
        key="wiki_off",
        use_wiki_context=False,
        wiki_doc_routing=False,
    ),
    "wiki_doc_routing": ConditionSpec(
        key="wiki_doc_routing",
        use_wiki_context=True,
        wiki_doc_routing=True,
    ),
    "wiki_static_refs": ConditionSpec(
        key="wiki_static_refs",
        use_wiki_context=True,
        wiki_doc_routing=False,
    ),
}


class RetryableCaseError(RuntimeError):
    """Raised when run_agentic_search returns an infrastructure-style error."""


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def contains_gold_answer(answer: str, gold: str) -> bool:
    """Return true when the normalized non-empty answer contains the gold answer."""
    normalized_answer = normalize_answer(answer)
    normalized_gold = normalize_answer(gold)
    return bool(normalized_answer) and bool(normalized_gold) and (
        normalized_gold in normalized_answer
    )


def _parse_keys(raw: str, available: dict[str, Any], label: str) -> list[str]:
    keys = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [key for key in keys if key not in available]
    if unknown:
        valid = ", ".join(sorted(available))
        raise SystemExit(f"Unknown {label}: {', '.join(unknown)}. Valid: {valid}")
    return keys


def _report_path(results_dir: Path, model: ModelSpec, condition: ConditionSpec, rw: int) -> Path:
    return results_dir / f"multihop_full_{model.key}_{condition.key}_rw{rw}.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"warning: ignoring invalid JSONL line {path}:{line_number}")
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _dedupe_latest_by_id(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        case_id = row.get("id")
        if not isinstance(case_id, str) or not case_id:
            continue
        if case_id not in by_id:
            order.append(case_id)
        by_id[case_id] = row
    return [by_id[case_id] for case_id in order]


def completed_case_ids(path: Path) -> set[str]:
    """Return case ids that already have a JSONL row."""
    return {row["id"] for row in _dedupe_latest_by_id(_read_jsonl(path))}


@contextmanager
def temporary_env(updates: dict[str, str]) -> Iterator[None]:
    """Temporarily patch process environment variables."""
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _env_for(model: ModelSpec, condition: ConditionSpec) -> dict[str, str]:
    return {
        "AGENT_LLM_PROVIDER": "ollama",
        "AGENT_LLM_MODEL": model.model,
        "AGENT_LLM_BASE_URL": model.base_url,
        "OLLAMA_BASE_URL": model.base_url,
        "WIKI_CONTEXT_INJECT": "false",
        "WIKI_DOC_ROUTING": "true" if condition.wiki_doc_routing else "false",
    }


async def _run_agent_case_once(  # noqa: PLR0913
    *,
    case: BenchmarkCase,
    collection_id: str,
    model: ModelSpec,
    condition: ConditionSpec,
    max_rewrites: int,
    search_type: str,
    search_limit: int,
    case_timeout: float,
) -> dict[str, Any]:
    async def invoke() -> dict[str, Any]:
        with temporary_env(_env_for(model, condition)):
            return await run_agentic_search(
                question=case.question,
                collection_id=collection_id,
                search_type=search_type,  # type: ignore[arg-type]
                search_limit=search_limit,
                max_rewrites=max_rewrites,
                llm_provider="ollama",
                llm_model=model.model,
                use_wiki_context=condition.use_wiki_context,
            )

    if case_timeout > 0:
        result = await asyncio.wait_for(invoke(), timeout=case_timeout)
    else:
        result = await invoke()

    error = result.get("error")
    if error and error != NO_CONTEXT_ERROR:
        raise RetryableCaseError(str(error))
    return result


def _row_from_result(  # noqa: PLR0913
    *,
    case: BenchmarkCase,
    model: ModelSpec,
    condition: ConditionSpec,
    max_rewrites: int,
    search_type: str,
    search_limit: int,
    result: dict[str, Any],
    retry_count: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    answer = result.get("generation") or ""
    relevant = result.get("relevant_documents") or []
    steps = result.get("steps") or []
    return {
        "id": case.id,
        "question_type": case.question_type,
        "model_key": model.key,
        "model": model.model,
        "base_url": model.base_url,
        "condition": condition.key,
        "max_rewrites": max_rewrites,
        "search_type": search_type,
        "search_limit": search_limit,
        "wiki_doc_routing": condition.wiki_doc_routing,
        "wiki_context_inject": False,
        "answer": answer,
        "answer_preview": answer[:300],
        "token_f1": token_f1(answer, case.answer),
        "contains": contains_gold_answer(answer, case.answer),
        "evidence_recall": evidence_recall(case.expected_document_keys, relevant),
        "wiki_status": result.get("wiki_context_status"),
        "wiki_promotion_status": result.get("wiki_promotion_status"),
        "wiki_source_ref_count": len(result.get("wiki_source_refs") or []),
        "wiki_promoted_document_count": len(result.get("wiki_promoted_document_ids") or []),
        "retrieved_document_count": len(result.get("retrieved_document_ids") or []),
        "wiki_injected": any(
            "injected into generation prompt" in str(step) for step in steps
        ),
        "error": result.get("error"),
        "retry_count": retry_count,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "completed_at": _utc_now(),
    }


def _row_from_failure(  # noqa: PLR0913
    *,
    case: BenchmarkCase,
    model: ModelSpec,
    condition: ConditionSpec,
    max_rewrites: int,
    search_type: str,
    search_limit: int,
    error: BaseException,
    retry_count: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "id": case.id,
        "question_type": case.question_type,
        "model_key": model.key,
        "model": model.model,
        "base_url": model.base_url,
        "condition": condition.key,
        "max_rewrites": max_rewrites,
        "search_type": search_type,
        "search_limit": search_limit,
        "wiki_doc_routing": condition.wiki_doc_routing,
        "wiki_context_inject": False,
        "answer": "",
        "answer_preview": "",
        "token_f1": 0.0,
        "contains": False,
        "evidence_recall": None,
        "wiki_status": None,
        "wiki_promotion_status": None,
        "wiki_source_ref_count": 0,
        "wiki_promoted_document_count": 0,
        "retrieved_document_count": 0,
        "wiki_injected": False,
        "error": str(error),
        "exception_type": type(error).__name__,
        "retry_count": retry_count,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "completed_at": _utc_now(),
    }


async def run_case_with_retries(  # noqa: PLR0913
    *,
    case: BenchmarkCase,
    collection_id: str,
    model: ModelSpec,
    condition: ConditionSpec,
    max_rewrites: int,
    search_type: str,
    search_limit: int,
    retry_limit: int,
    case_timeout: float,
) -> dict[str, Any]:
    """Run one case, retry transient failures, and return a JSONL row."""
    started = time.perf_counter()
    last_error: BaseException | None = None
    for retry_count in range(retry_limit + 1):
        try:
            result = await _run_agent_case_once(
                case=case,
                collection_id=collection_id,
                model=model,
                condition=condition,
                max_rewrites=max_rewrites,
                search_type=search_type,
                search_limit=search_limit,
                case_timeout=case_timeout,
            )
            return _row_from_result(
                case=case,
                model=model,
                condition=condition,
                max_rewrites=max_rewrites,
                search_type=search_type,
                search_limit=search_limit,
                result=result,
                retry_count=retry_count,
                elapsed_seconds=time.perf_counter() - started,
            )
        except Exception as exc:
            last_error = exc
            if retry_count >= retry_limit:
                break
            print(
                f"retry {retry_count + 1}/{retry_limit}: "
                f"{model.key}/{condition.key}/{case.id}: {exc}",
                flush=True,
            )

    if last_error is None:
        msg = "case failed without recording an exception"
        raise RuntimeError(msg)
    return _row_from_failure(
        case=case,
        model=model,
        condition=condition,
        max_rewrites=max_rewrites,
        search_type=search_type,
        search_limit=search_limit,
        error=last_error,
        retry_count=retry_limit,
        elapsed_seconds=time.perf_counter() - started,
    )


def append_row(path: Path, row: dict[str, Any]) -> None:
    """Append one benchmark row to a JSONL report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate deterministic benchmark metrics from JSONL rows."""
    rows = _dedupe_latest_by_id(rows)
    n = len(rows)
    recalls = [row["evidence_recall"] for row in rows if row.get("evidence_recall") is not None]
    return {
        "case_count": n,
        "error_count": sum(1 for row in rows if row.get("error")),
        "no_relevant_context_count": sum(
            1 for row in rows if row.get("error") == NO_CONTEXT_ERROR
        ),
        "exception_count": sum(1 for row in rows if row.get("exception_type")),
        "contains_count": sum(1 for row in rows if row.get("contains")),
        "mean_token_f1": (
            sum(float(row.get("token_f1") or 0.0) for row in rows) / n if n else None
        ),
        "mean_evidence_recall": (
            sum(float(value) for value in recalls) / len(recalls) if recalls else None
        ),
        "wiki_injected_count": sum(1 for row in rows if row.get("wiki_injected")),
        "retry_count_total": sum(int(row.get("retry_count") or 0) for row in rows),
    }


def write_summary(
    *,
    path: Path,
    results_dir: Path,
    models: list[ModelSpec],
    conditions: list[ConditionSpec],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Write the cross-model/condition summary JSON."""
    entries: dict[str, Any] = {}
    for model in models:
        for condition in conditions:
            report = _report_path(results_dir, model, condition, args.max_rewrites)
            rows = _read_jsonl(report)
            entries[f"{model.key}/{condition.key}"] = {
                "report_path": str(report),
                **summarize_rows(rows),
            }

    summary = {
        "metadata": {
            "generated_at": _utc_now(),
            "collection_id": args.collection_id,
            "dataset": args.dataset,
            "results_dir": str(results_dir),
            "max_rewrites": args.max_rewrites,
            "retry_limit": args.retry_limit,
            "case_timeout": args.case_timeout,
            "limit": args.limit,
            "search_type": args.search_type,
            "search_limit": args.search_limit,
            "wiki_context_inject": False,
        },
        "results": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


async def run_matrix(args: argparse.Namespace) -> int:
    """Run the selected model/condition matrix with resumable JSONL outputs."""
    results_dir = Path(args.results_dir)
    model_specs = [MODELS[key] for key in _parse_keys(args.models, MODELS, "model")]
    condition_specs = [
        CONDITIONS[key] for key in _parse_keys(args.conditions, CONDITIONS, "condition")
    ]

    cases = load_cases(Path(args.dataset))
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be at least 1")
        cases = cases[: args.limit]

    print(
        f"cases={len(cases)} models={[model.key for model in model_specs]} "
        f"conditions={[condition.key for condition in condition_specs]} "
        f"max_rewrites={args.max_rewrites}",
        flush=True,
    )

    for model in model_specs:
        for condition in condition_specs:
            report = _report_path(results_dir, model, condition, args.max_rewrites)
            done = completed_case_ids(report)
            pending = [case for case in cases if case.id not in done]
            print(
                f"\n== {model.key} / {condition.key} == "
                f"done={len(done)} pending={len(pending)} report={report}",
                flush=True,
            )
            for offset, case in enumerate(pending, start=1):
                row = await run_case_with_retries(
                    case=case,
                    collection_id=args.collection_id,
                    model=model,
                    condition=condition,
                    max_rewrites=args.max_rewrites,
                    search_type=args.search_type,
                    search_limit=args.search_limit,
                    retry_limit=args.retry_limit,
                    case_timeout=args.case_timeout,
                )
                append_row(report, row)
                completed = len(done) + offset
                if completed % args.progress_every == 0 or completed == len(cases):
                    print(
                        f"  ...{completed}/{len(cases)} "
                        f"contains={row['contains']} error={row.get('error')}",
                        flush=True,
                    )

    summary_path = Path(args.summary_path) if args.summary_path else (
        results_dir / f"multihop_full_summary_rw{args.max_rewrites}.json"
    )
    summary = write_summary(
        path=summary_path,
        results_dir=results_dir,
        models=model_specs,
        conditions=condition_specs,
        args=args,
    )
    print("\n=== Summary ===")
    for key, value in summary["results"].items():
        print(
            f"{key}: n={value['case_count']} contains={value['contains_count']} "
            f"errors={value['error_count']} recall={value['mean_evidence_recall']}",
            flush=True,
        )
    print(f"summary: {summary_path}", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the benchmark runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--summary-path")
    parser.add_argument("--max-rewrites", type=int, default=DEFAULT_MAX_REWRITES)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--retry-limit", type=int, default=DEFAULT_RETRY_LIMIT)
    parser.add_argument("--case-timeout", type=float, default=DEFAULT_CASE_TIMEOUT_SECONDS)
    parser.add_argument("--search-type", default=DEFAULT_SEARCH_TYPE)
    parser.add_argument("--search-limit", type=int, default=DEFAULT_SEARCH_LIMIT)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args(argv)


async def _main(argv: list[str] | None = None) -> int:
    try:
        return await run_matrix(parse_args(argv))
    finally:
        await close_db_pool()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
