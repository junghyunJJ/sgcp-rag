"""Benchmark SGCP-RAG hybrid search ranking through the public REST API."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx

DEFAULT_API_BASE = "http://localhost:8888"
DEFAULT_COLLECTION_NAME = "agentpaper"
DEFAULT_LIMIT = 8
DEFAULT_CANDIDATE_LIMIT = 50
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_TOP_N = 3
MIN_SCORE_EXPLORATORY = 0.70
DEFAULT_OFFLINE_WEIGHTS = (0.7, 0.6, 0.5)

Expectation = Literal[
    "top_source",
    "source_in_top_n",
    "empty",
    "filter_all_match",
    "non_empty",
]

ALLOWED_EXPECTATIONS: set[str] = {
    "top_source",
    "source_in_top_n",
    "empty",
    "filter_all_match",
    "non_empty",
}


class BenchmarkConfigError(RuntimeError):
    """Raised when benchmark configuration cannot be resolved safely."""


@dataclass(frozen=True)
class BenchmarkCase:
    """A source-level search expectation for one query."""

    name: str
    query: str
    expectation: Expectation
    filter: dict[str, Any] | None = None
    expected_source: str | None = None
    expected_sources: list[str] = field(default_factory=list)
    top_n: int = DEFAULT_TOP_N
    expected_empty: bool = False
    warning_if_expected_rank_gt: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class CaseEvaluation:
    """Evaluation summary for one case and one benchmark lane."""

    case_name: str
    lane: str
    passed: bool
    warning: bool
    result_count: int
    expected_rank: int | None
    top_source: str | None
    top_score: float | None
    top_sources: list[str]
    message: str = ""


@dataclass(frozen=True)
class LaneResult:
    """Search rows and evaluation for one lane."""

    lane: str
    results: list[dict[str, Any]]
    evaluation: CaseEvaluation


@dataclass(frozen=True)
class ApiContext:
    """REST API context for benchmark requests."""

    client: httpx.Client
    api_base: str
    collection_id: str


def _default_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            name="agent-skill",
            query="agent skill",
            expectation="source_in_top_n",
            expected_sources=["skillfoundry.pdf", "SkillClaw.pdf"],
            top_n=DEFAULT_TOP_N,
        ),
        BenchmarkCase(
            name="reusable-skill-libraries",
            query="reusable skill libraries lifelong embodied learning",
            expectation="top_source",
            expected_source="skillfoundry.pdf",
        ),
        BenchmarkCase(
            name="voyager-executable-skills",
            query="Voyager executable skills",
            expectation="source_in_top_n",
            expected_source="skillfoundry.pdf",
            top_n=DEFAULT_TOP_N,
            warning_if_expected_rank_gt=1,
        ),
        BenchmarkCase(
            name="anthropic-claude-agent-skills",
            query="Anthropic Claude framework formalized agent skills",
            expectation="source_in_top_n",
            expected_source="skillfoundry.pdf",
            top_n=DEFAULT_TOP_N,
            warning_if_expected_rank_gt=1,
        ),
        BenchmarkCase(
            name="conservative-editing-mode",
            query="Conservative editing mode",
            expectation="source_in_top_n",
            expected_source="SkillClaw.pdf",
            top_n=DEFAULT_TOP_N,
        ),
        BenchmarkCase(
            name="create-skill-schema",
            query="create_skill rationale new-lowercase-slug",
            expectation="top_source",
            expected_source="SkillClaw.pdf",
        ),
        BenchmarkCase(
            name="sop-like-guidance",
            query="SOP-like guidance agent behavior",
            expectation="top_source",
            expected_source="SkillClaw.pdf",
        ),
        BenchmarkCase(
            name="web-skill-induction",
            query="web skill induction",
            expectation="source_in_top_n",
            expected_source="SkillClaw.pdf",
            top_n=DEFAULT_TOP_N,
        ),
        BenchmarkCase(
            name="spatial-transcriptomics-reference",
            query="spatial transcriptomics transfer learning",
            expectation="source_in_top_n",
            expected_source="skillfoundry.pdf",
            top_n=DEFAULT_TOP_N,
        ),
        BenchmarkCase(
            name="nonsense-empty",
            query="zzzz qwerty nonexistent agentpaper noise",
            expectation="empty",
            expected_empty=True,
        ),
        BenchmarkCase(
            name="filter-skillfoundry",
            query="agent skill",
            expectation="filter_all_match",
            expected_source="skillfoundry.pdf",
            filter={"source": "skillfoundry.pdf"},
        ),
        BenchmarkCase(
            name="filter-skillclaw",
            query="agent skill",
            expectation="filter_all_match",
            expected_source="SkillClaw.pdf",
            filter={"source": "SkillClaw.pdf"},
        ),
    ]


def _source_of(result: dict[str, Any]) -> str | None:
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        return None
    source = metadata.get("source")
    return str(source) if source is not None else None


def _score_of(result: dict[str, Any]) -> float | None:
    score = result.get("score")
    if isinstance(score, int | float):
        return float(score)
    return None


def _string_id(result: dict[str, Any]) -> str:
    return str(result.get("id"))


def _candidate_sources(case: BenchmarkCase) -> list[str]:
    sources = list(case.expected_sources)
    if case.expected_source:
        sources.append(case.expected_source)
    return sources


def _rank_for_sources(
    results: list[dict[str, Any]],
    expected_sources: list[str],
) -> int | None:
    for index, result in enumerate(results, start=1):
        if _source_of(result) in expected_sources:
            return index
    return None


def _top_sources(results: list[dict[str, Any]], limit: int = DEFAULT_TOP_N) -> list[str]:
    return [
        source
        for source in (_source_of(result) for result in results[:limit])
        if source is not None
    ]


def _validate_case(case: BenchmarkCase) -> None:
    if case.expectation not in ALLOWED_EXPECTATIONS:
        raise BenchmarkConfigError(f"Unsupported expectation: {case.expectation}")
    if case.top_n < 1:
        raise BenchmarkConfigError(f"Case {case.name!r} top_n must be at least 1")
    if case.expectation in {
        "top_source",
        "source_in_top_n",
        "filter_all_match",
    } and not _candidate_sources(case):
        raise BenchmarkConfigError(
            f"Case {case.name!r} requires expected_source or expected_sources"
        )


def _case_from_mapping(raw_case: dict[str, Any]) -> BenchmarkCase:
    expectation = raw_case.get("expectation")
    if expectation not in ALLOWED_EXPECTATIONS:
        raise BenchmarkConfigError(f"Unsupported expectation: {expectation}")
    case = BenchmarkCase(
        name=str(raw_case["name"]),
        query=str(raw_case["query"]),
        expectation=expectation,  # type: ignore[arg-type]
        filter=raw_case.get("filter"),
        expected_source=raw_case.get("expected_source"),
        expected_sources=list(raw_case.get("expected_sources") or []),
        top_n=int(raw_case.get("top_n", DEFAULT_TOP_N)),
        expected_empty=bool(raw_case.get("expected_empty", False)),
        warning_if_expected_rank_gt=raw_case.get("warning_if_expected_rank_gt"),
        notes=str(raw_case.get("notes", "")),
    )
    _validate_case(case)
    return case


def load_cases(path: Path | None) -> list[BenchmarkCase]:
    """Load benchmark cases from JSON or return built-in cases."""
    if path is None:
        cases = _default_cases()
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise BenchmarkConfigError("Cases file must contain a JSON list")
        cases = [_case_from_mapping(item) for item in raw]
    for case in cases:
        _validate_case(case)
    return cases


def resolve_collection_id(
    collections: list[dict[str, Any]],
    collection_name: str,
) -> str:
    """Resolve a collection name to a unique UUID."""
    matches = [
        collection
        for collection in collections
        if collection.get("name") == collection_name
    ]
    if not matches:
        raise BenchmarkConfigError(f"No collection named {collection_name!r} found")
    if len(matches) > 1:
        details = ", ".join(
            (
                f"{collection.get('uuid')} "
                f"(docs={collection.get('document_count')}, "
                f"chunks={collection.get('chunk_count')})"
            )
            for collection in matches
        )
        raise BenchmarkConfigError(
            f"Multiple collections named {collection_name!r}: {details}. "
            "Pass --collection-id."
        )
    uuid = matches[0].get("uuid")
    if not uuid:
        raise BenchmarkConfigError(f"Collection {collection_name!r} has no uuid")
    return str(uuid)


def evaluate_results(
    case: BenchmarkCase,
    lane: str,
    results: list[dict[str, Any]],
) -> CaseEvaluation:
    """Evaluate one lane's results against a benchmark case."""
    expected_sources = _candidate_sources(case)
    expected_rank = _rank_for_sources(results, expected_sources)
    top_source = _source_of(results[0]) if results else None
    top_score = _score_of(results[0]) if results else None
    warning = False
    message = ""

    if case.expectation == "empty":
        passed = len(results) == 0
        if not passed:
            message = "expected empty results"
    elif case.expectation == "non_empty":
        passed = len(results) > 0
        if not passed:
            message = "expected at least one result"
    elif case.expectation == "top_source":
        passed = expected_rank == 1
        if not passed:
            message = f"expected top source in {expected_sources}"
    elif case.expectation == "source_in_top_n":
        passed = expected_rank is not None and expected_rank <= case.top_n
        if not passed:
            message = f"expected source in top {case.top_n}: {expected_sources}"
    elif case.expectation == "filter_all_match":
        passed = all(_source_of(result) in expected_sources for result in results)
        if not passed:
            message = f"expected all sources in {expected_sources}"
    else:
        raise BenchmarkConfigError(f"Unsupported expectation: {case.expectation}")

    if (
        case.warning_if_expected_rank_gt is not None
        and expected_rank is not None
        and expected_rank > case.warning_if_expected_rank_gt
    ):
        warning = True
        message = (
            f"{message}; " if message else ""
        ) + f"expected source rank {expected_rank}"

    return CaseEvaluation(
        case_name=case.name,
        lane=lane,
        passed=passed,
        warning=warning,
        result_count=len(results),
        expected_rank=expected_rank,
        top_source=top_source,
        top_score=top_score,
        top_sources=_top_sources(results),
        message=message,
    )


def fuse_results(
    semantic_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
    *,
    semantic_weight: float,
) -> list[dict[str, Any]]:
    """Fuse semantic and keyword rows by string id for offline simulation."""
    keyword_weight = 1.0 - semantic_weight
    combined: dict[str, dict[str, Any]] = {}
    max_keyword_score = max(
        (float(result.get("score", 0.0)) for result in keyword_results),
        default=0.0,
    )

    for result in semantic_results:
        result_id = _string_id(result)
        row = dict(result)
        row["id"] = result_id
        row["semantic_score"] = float(result.get("score", 0.0))
        row["keyword_score"] = 0.0
        combined[result_id] = row

    for result in keyword_results:
        result_id = _string_id(result)
        raw_keyword_score = float(result.get("score", 0.0))
        keyword_score = (
            raw_keyword_score / max_keyword_score
            if max_keyword_score > 0
            else 0.0
        )
        if result_id in combined:
            combined[result_id]["keyword_score"] = keyword_score
        else:
            row = dict(result)
            row["id"] = result_id
            row["semantic_score"] = 0.0
            row["keyword_score"] = keyword_score
            combined[result_id] = row

    fused = []
    for row in combined.values():
        semantic_score = float(row["semantic_score"])
        keyword_score = float(row["keyword_score"])
        fused_row = {
            key: value
            for key, value in row.items()
            if key not in {"semantic_score", "keyword_score"}
        }
        fused_row["score"] = (
            semantic_score * semantic_weight + keyword_score * keyword_weight
        )
        fused.append(fused_row)

    return sorted(fused, key=lambda item: float(item.get("score", 0.0)), reverse=True)


def exit_code_for(
    evaluations: list[CaseEvaluation],
    *,
    fail_on_regression: bool,
) -> int:
    """Return process exit code for expectation results."""
    if not fail_on_regression:
        return 0
    return (
        1
        if any(
            evaluation.lane == "current_hybrid" and not evaluation.passed
            for evaluation in evaluations
        )
        else 0
    )


def _get_json(client: httpx.Client, url: str) -> object:
    response = client.get(url)
    response.raise_for_status()
    return response.json()


def _post_json(client: httpx.Client, url: str, payload: dict[str, Any]) -> object:
    response = client.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def _fetch_collections(client: httpx.Client, api_base: str) -> list[dict[str, Any]]:
    data = _get_json(client, f"{api_base}/collections")
    if not isinstance(data, list):
        raise BenchmarkConfigError("GET /collections did not return a list")
    return data


def _search(
    context: ApiContext,
    case: BenchmarkCase,
    *,
    search_type: str,
    limit: int,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    payload: dict[str, Any] = {
        "query": case.query,
        "limit": limit,
        "search_type": search_type,
    }
    if case.filter is not None:
        payload["filter"] = case.filter
    if min_score is not None:
        payload["min_score"] = min_score

    data = _post_json(
        context.client,
        f"{context.api_base}/collections/"
        f"{context.collection_id}/documents/search",
        payload,
    )
    if not isinstance(data, list):
        raise BenchmarkConfigError("Search endpoint did not return a list")
    return data


def _benchmark_case(
    context: ApiContext,
    case: BenchmarkCase,
    *,
    limit: int,
    candidate_limit: int,
) -> list[LaneResult]:
    semantic = _search(
        context,
        case,
        search_type="semantic",
        limit=limit,
    )
    keyword = _search(
        context,
        case,
        search_type="keyword",
        limit=limit,
    )
    current_hybrid = _search(
        context,
        case,
        search_type="hybrid",
        limit=limit,
    )
    current_hybrid_min_070 = _search(
        context,
        case,
        search_type="hybrid",
        limit=limit,
        min_score=MIN_SCORE_EXPLORATORY,
    )

    lane_rows = {
        "semantic": semantic,
        "keyword": keyword,
        "current_hybrid": current_hybrid,
        "current_hybrid_min_070": current_hybrid_min_070,
    }

    semantic_candidates = _search(
        context,
        case,
        search_type="semantic",
        limit=candidate_limit,
        min_score=0.0,
    )
    keyword_candidates = _search(
        context,
        case,
        search_type="keyword",
        limit=candidate_limit,
    )
    for semantic_weight in DEFAULT_OFFLINE_WEIGHTS:
        lane_name = f"offline_fusion_{semantic_weight:.1f}_{1 - semantic_weight:.1f}"
        lane_rows[lane_name] = fuse_results(
            semantic_candidates,
            keyword_candidates,
            semantic_weight=semantic_weight,
        )[:limit]

    return [
        LaneResult(
            lane=lane,
            results=results,
            evaluation=evaluate_results(case, lane, results),
        )
        for lane, results in lane_rows.items()
    ]


def _keyword_top_rank(lanes: list[LaneResult]) -> int | None:
    keyword = next((lane for lane in lanes if lane.lane == "keyword"), None)
    hybrid = next((lane for lane in lanes if lane.lane == "current_hybrid"), None)
    if keyword is None or hybrid is None or not keyword.results:
        return None
    keyword_top_id = _string_id(keyword.results[0])
    hybrid_ids = [_string_id(result) for result in hybrid.results]
    if keyword_top_id not in hybrid_ids:
        return None
    return hybrid_ids.index(keyword_top_id) + 1


def _snippet(result: dict[str, Any], max_len: int = 180) -> str:
    content = " ".join(str(result.get("page_content", "")).split())
    return content[:max_len]


def _serialize_lane(lane: LaneResult) -> dict[str, Any]:
    return {
        "lane": lane.lane,
        "evaluation": asdict(lane.evaluation),
        "results": [
            {
                "id": result.get("id"),
                "source": _source_of(result),
                "score": _score_of(result),
                "snippet": _snippet(result),
            }
            for result in lane.results
        ],
    }


def _serialize_report(
    cases: list[BenchmarkCase],
    results_by_case: dict[str, list[LaneResult]],
    *,
    collection_id: str,
) -> dict[str, Any]:
    evaluations = [
        lane.evaluation
        for lanes in results_by_case.values()
        for lane in lanes
    ]
    warnings = [evaluation for evaluation in evaluations if evaluation.warning]
    failures = [evaluation for evaluation in evaluations if not evaluation.passed]
    current_hybrid = [
        evaluation
        for evaluation in evaluations
        if evaluation.lane == "current_hybrid"
    ]
    current_hybrid_failures = [
        evaluation for evaluation in current_hybrid if not evaluation.passed
    ]
    keyword_top_late = 0
    for lanes in results_by_case.values():
        rank = _keyword_top_rank(lanes)
        if rank is None or rank > 1:
            keyword_top_late += 1

    return {
        "collection_id": collection_id,
        "summary": {
            "case_count": len(cases),
            "evaluation_count": len(evaluations),
            "pass_count": len(evaluations) - len(failures),
            "failure_count": len(failures),
            "warning_count": len(warnings),
            "current_hybrid_pass_count": len(current_hybrid)
            - len(current_hybrid_failures),
            "current_hybrid_failure_count": len(current_hybrid_failures),
            "keyword_top_rank_gt_1_or_absent": keyword_top_late,
        },
        "cases": [
            {
                "case": asdict(case),
                "keyword_top_rank_in_current_hybrid": _keyword_top_rank(
                    results_by_case[case.name]
                ),
                "lanes": [
                    _serialize_lane(lane)
                    for lane in results_by_case[case.name]
                ],
            }
            for case in cases
        ],
        "notes": [
            "current_hybrid uses the public production API with --limit.",
            (
                "offline_fusion lanes are approximate simulations built from public "
                "semantic and keyword API responses."
            ),
        ],
    }


def _status_text(value: object) -> str:
    return "PASS" if value else "FAIL"


def _format_table(report: dict[str, Any]) -> str:
    lines = [
        (
            "case | lane | pass | warn | count | expected_rank | "
            "top_source | top_score"
        ),
        "--- | --- | --- | --- | ---: | ---: | --- | ---:",
    ]
    for case_entry in report["cases"]:
        case_name = case_entry["case"]["name"]
        for lane in case_entry["lanes"]:
            evaluation = lane["evaluation"]
            top_score = evaluation["top_score"]
            top_score_text = "" if top_score is None else f"{top_score:.4f}"
            expected_rank = evaluation["expected_rank"]
            expected_rank_text = "" if expected_rank is None else str(expected_rank)
            lines.append(
                " | ".join(
                    [
                        case_name,
                        evaluation["lane"],
                        _status_text(evaluation["passed"]),
                        "WARN" if evaluation["warning"] else "",
                        str(evaluation["result_count"]),
                        expected_rank_text,
                        str(evaluation["top_source"] or ""),
                        top_score_text,
                    ]
                )
            )
    summary = report["summary"]
    lines.extend([
        "",
        (
            f"Summary: {summary['pass_count']}/{summary['evaluation_count']} "
            f"passed, {summary['warning_count']} warnings, "
            f"{summary['failure_count']} failures"
        ),
        (
            "Current hybrid gate: "
            f"{summary['current_hybrid_pass_count']}/"
            f"{summary['case_count']} passed, "
            f"{summary['current_hybrid_failure_count']} failures"
        ),
        (
            "Keyword top rank >1 or absent in current_hybrid: "
            f"{summary['keyword_top_rank_gt_1_or_absent']}"
        ),
    ])
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SGCP-RAG hybrid search ranking via REST.",
    )
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--collection-id")
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=DEFAULT_CANDIDATE_LIMIT,
    )
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--format", choices=["table", "json"], default="table")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-regression", action="store_true")
    return parser.parse_args(argv)


def run_benchmark(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    """Run the benchmark and return its report plus process exit code."""
    if args.limit < 1:
        raise BenchmarkConfigError("--limit must be at least 1")
    if args.candidate_limit < args.limit:
        raise BenchmarkConfigError("--candidate-limit must be >= --limit")

    api_base = str(args.api_base).rstrip("/")
    cases = load_cases(args.cases)
    with httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
        if args.collection_id:
            collection_id = str(args.collection_id)
        else:
            collections = _fetch_collections(client, api_base)
            collection_id = resolve_collection_id(collections, args.collection_name)

        results_by_case = {
            case.name: _benchmark_case(
                ApiContext(
                    client=client,
                    api_base=api_base,
                    collection_id=collection_id,
                ),
                case,
                limit=args.limit,
                candidate_limit=args.candidate_limit,
            )
            for case in cases
        }

    report = _serialize_report(cases, results_by_case, collection_id=collection_id)
    evaluations = [
        LaneResult(
            lane=lane["lane"],
            results=[],
            evaluation=CaseEvaluation(**lane["evaluation"]),
        ).evaluation
        for case_entry in report["cases"]
        for lane in case_entry["lanes"]
    ]
    return report, exit_code_for(
        evaluations,
        fail_on_regression=bool(args.fail_on_regression),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    try:
        report, exit_code = run_benchmark(args)
    except (BenchmarkConfigError, httpx.HTTPError) as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        output = json.dumps(report, ensure_ascii=False, indent=2)
    else:
        output = _format_table(report)

    if args.output is not None:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
