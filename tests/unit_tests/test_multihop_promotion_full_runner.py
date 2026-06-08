from __future__ import annotations

import argparse
import asyncio
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from benchmarking.scripts import run_multihop_promotion_full as runner


def _write_cases(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": "case-1",
                    "query": "first question",
                    "answer": "alpha",
                    "question_type": "comparison",
                    "evidence_list": [{"url": "doc-a"}],
                },
                {
                    "id": "case-2",
                    "query": "second question",
                    "answer": "beta",
                    "question_type": "bridge",
                    "evidence_list": [{"url": "doc-b"}],
                },
            ]
        ),
        encoding="utf-8",
    )


def _args(tmp_path: Path, cases: Path, **overrides: object) -> argparse.Namespace:
    values = {
        "collection_id": "collection-id",
        "dataset": str(cases),
        "results_dir": str(tmp_path / "results"),
        "summary_path": str(tmp_path / "summary.json"),
        "max_rewrites": 3,
        "limit": None,
        "models": "qwen35_122b_port6000",
        "conditions": "wiki_off",
        "retry_limit": 2,
        "case_timeout": 0.0,
        "search_type": "hybrid",
        "search_limit": 5,
        "progress_every": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_contains_requires_non_empty_answer() -> None:
    """Empty answers must not pass containment scoring."""
    assert not runner.contains_gold_answer("", "alpha")
    assert runner.contains_gold_answer("The answer is alpha.", "alpha")


def test_run_matrix_resumes_from_existing_jsonl(tmp_path: Path, monkeypatch) -> None:
    """Existing case ids are skipped and not duplicated on resume."""
    cases = tmp_path / "cases.json"
    _write_cases(cases)

    results_dir = tmp_path / "results"
    report = (
        results_dir
        / "multihop_full_qwen35_122b_port6000_wiki_off_rw3.jsonl"
    )
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps({"id": "case-1", "contains": False}) + "\n",
        encoding="utf-8",
    )

    calls: list[str] = []

    async def fake_run_agentic_search(**kwargs: object) -> dict[str, object]:
        calls.append(str(kwargs["question"]))
        return {
            "generation": "beta",
            "relevant_documents": [{"metadata": {"url": "doc-b"}}],
            "steps": [],
            "error": None,
            "wiki_context_status": "disabled",
            "wiki_promotion_status": "disabled",
            "wiki_source_refs": [],
            "wiki_promoted_document_ids": [],
            "retrieved_document_ids": ["doc-b"],
        }

    monkeypatch.setattr(runner, "run_agentic_search", fake_run_agentic_search)

    asyncio.run(runner.run_matrix(_args(tmp_path, cases)))

    rows = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
    assert calls == ["second question"]
    assert [row["id"] for row in rows] == ["case-1", "case-2"]
    assert rows[1]["contains"] is True
    assert rows[1]["evidence_recall"] == 1.0

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    key = "qwen35_122b_port6000/wiki_off"
    assert summary["results"][key]["case_count"] == 2
    assert summary["results"][key]["contains_count"] == 1


def test_run_case_retries_transient_failure(tmp_path: Path, monkeypatch) -> None:
    """Transient exceptions are retried before a row is emitted."""
    cases = tmp_path / "cases.json"
    _write_cases(cases)
    case = runner.load_cases(cases)[0]
    model = runner.MODELS["qwen35_122b_port6000"]
    condition = runner.CONDITIONS["wiki_off"]
    attempts = 0

    async def fake_run_agentic_search(**_kwargs: object) -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary failure")
        return {
            "generation": "alpha",
            "relevant_documents": [{"metadata": {"url": "doc-a"}}],
            "steps": [],
            "error": None,
            "wiki_context_status": "disabled",
            "wiki_promotion_status": "disabled",
            "wiki_source_refs": [],
            "wiki_promoted_document_ids": [],
            "retrieved_document_ids": ["doc-a"],
        }

    monkeypatch.setattr(runner, "run_agentic_search", fake_run_agentic_search)

    row = asyncio.run(
        runner.run_case_with_retries(
            case=case,
            collection_id="collection-id",
            model=model,
            condition=condition,
            max_rewrites=3,
            search_type="hybrid",
            search_limit=5,
            retry_limit=2,
            case_timeout=0.0,
        )
    )

    assert attempts == 2
    assert row["retry_count"] == 1
    assert row["contains"] is True
