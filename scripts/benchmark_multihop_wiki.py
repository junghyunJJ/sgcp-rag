"""Benchmark agentic RAG answers with wiki off/on lanes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx

DEFAULT_API_BASE = "http://localhost:8888"
DEFAULT_DATASET_NAME = "MultiHop-RAG"
DEFAULT_PILOT_SIZE = 50
DEFAULT_SEED = 13
DEFAULT_SEARCH_TYPE = "hybrid"
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_MAX_REWRITES = 3
DEFAULT_TIMEOUT_SECONDS = 300.0
JUDGE_SNIPPET_CHARS = 800

LaneName = Literal["wiki_off", "wiki_on"]


class BenchmarkConfigError(RuntimeError):
    """Raised when benchmark inputs or options are invalid."""


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark question plus gold answer and evidence keys."""

    id: str
    question: str
    answer: str
    question_type: str | None = None
    expected_document_keys: list[str] = field(default_factory=list)
    expected_facts: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    """Frozen request parameters shared by both benchmark lanes."""

    collection_id: str
    search_type: Literal["semantic", "keyword", "hybrid"] = DEFAULT_SEARCH_TYPE
    search_limit: int = DEFAULT_SEARCH_LIMIT
    max_rewrites: int = DEFAULT_MAX_REWRITES
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_temperature: float | None = None
    min_score: float | None = None
    search_filter: dict[str, Any] | None = None


@dataclass(frozen=True)
class LaneResult:
    """REST result for one case/lane."""

    lane: LaneName
    payload: dict[str, Any]
    response: dict[str, Any]
    latency_seconds: float
    error: str | None = None


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkConfigError(f"Invalid JSON file: {path}") from exc


def _records_from_json(data: object) -> list[dict[str, Any]]:
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        for key in ("data", "records", "queries", "test", "train", "validation"):
            if isinstance(data.get(key), list):
                records = data[key]
                break
        else:
            if all(isinstance(value, dict) for value in data.values()):
                records = list(data.values())
            else:
                raise BenchmarkConfigError("JSON object does not contain case records")
    else:
        raise BenchmarkConfigError("Benchmark cases must be a JSON list or object")

    if not all(isinstance(record, dict) for record in records):
        raise BenchmarkConfigError("Every benchmark case must be a JSON object")
    return records


def _require_text(raw: dict[str, Any], keys: tuple[str, ...], *, field: str) -> str:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise BenchmarkConfigError(f"Case is missing required {field!r} text")


def _dedupe_texts(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = value.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _evidence_keys_and_facts(raw: dict[str, Any]) -> tuple[list[str], list[str]]:
    evidence = raw.get("evidence_list") or raw.get("evidence") or []
    keys: list[str] = []
    facts: list[str] = []
    if not isinstance(evidence, list):
        return [], []

    for item in evidence:
        if isinstance(item, str):
            keys.append(item)
            continue
        if not isinstance(item, dict):
            continue
        for key in ("url", "title", "source", "doc_id", "document_id"):
            value = item.get(key)
            if isinstance(value, str):
                keys.append(value)
        fact = item.get("fact")
        if isinstance(fact, str) and fact.strip():
            facts.append(fact.strip())
    return _dedupe_texts(keys), _dedupe_texts(facts)


def load_cases(path: Path) -> list[BenchmarkCase]:
    """Load MultiHop-RAG style cases from JSON."""
    records = _records_from_json(_read_json(path))
    cases: list[BenchmarkCase] = []
    for index, raw in enumerate(records, start=1):
        question = _require_text(raw, ("query", "question"), field="question")
        answer = _require_text(raw, ("answer", "gold_answer"), field="answer")
        case_id = str(
            raw.get("id")
            or raw.get("query_id")
            or raw.get("_id")
            or f"case-{index:04d}"
        )
        question_type = raw.get("question_type")
        evidence_keys, facts = _evidence_keys_and_facts(raw)
        cases.append(
            BenchmarkCase(
                id=case_id,
                question=question,
                answer=answer,
                question_type=question_type if isinstance(question_type, str) else None,
                expected_document_keys=evidence_keys,
                expected_facts=facts,
                raw=raw,
            )
        )
    return cases


def select_pilot_cases(
    cases: list[BenchmarkCase],
    *,
    pilot_size: int = DEFAULT_PILOT_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[BenchmarkCase]:
    """Return a deterministic random pilot subset."""
    if pilot_size < 1:
        raise BenchmarkConfigError("--pilot-size must be at least 1")
    if pilot_size > len(cases):
        raise BenchmarkConfigError(
            f"Requested {pilot_size} cases, but only {len(cases)} are available"
        )
    return sorted(
        cases,
        key=lambda case: hashlib.sha256(f"{seed}:{case.id}".encode()).hexdigest(),
    )[:pilot_size]


def write_case_ids(cases: list[BenchmarkCase], path: Path) -> None:
    """Persist selected pilot ids for reproducible reruns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"case_ids": [case.id for case in cases]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_case_ids(path: Path) -> list[str]:
    """Load selected case ids from a previous pilot selection."""
    data = _read_json(path)
    ids = data.get("case_ids") if isinstance(data, dict) else data
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise BenchmarkConfigError("Case id file must be a list or {case_ids: [...]}")
    return ids


def select_cases_by_id(
    cases: list[BenchmarkCase],
    case_ids: list[str],
) -> list[BenchmarkCase]:
    """Select cases in the exact order specified by case_ids."""
    by_id = {case.id: case for case in cases}
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise BenchmarkConfigError(f"Unknown case ids: {', '.join(missing[:5])}")
    return [by_id[case_id] for case_id in case_ids]


def normalize_answer(value: str) -> str:
    """Normalize an answer for simple exact/F1/containment metrics."""
    lowered = value.lower()
    no_punct = lowered.translate(str.maketrans(dict.fromkeys(string.punctuation, " ")))
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punct)
    return " ".join(no_articles.split())


def token_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 after answer normalization."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _document_keys(document: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    for field_name in (
        "multihop_url",
        "url",
        "source_url",
        "multihop_title",
        "title",
        "document_title",
        "multihop_source",
        "source",
        "file_id",
        "doc_id",
        "document_id",
    ):
        value = metadata.get(field_name)
        if isinstance(value, str) and value.strip():
            keys.add(value.strip())
    doc_id = document.get("id")
    if isinstance(doc_id, str) and doc_id.strip():
        keys.add(doc_id.strip())
    return keys


def evidence_recall(
    expected_document_keys: list[str],
    relevant_documents: list[dict[str, Any]],
) -> float | None:
    """Compute recall of expected evidence keys in relevant documents."""
    expected = set(expected_document_keys)
    if not expected:
        return None
    observed: set[str] = set()
    for document in relevant_documents:
        observed.update(_document_keys(document))
    return len(expected & observed) / len(expected)


def evaluate_deterministic_metrics(
    answer: str,
    case: BenchmarkCase,
    relevant_documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute deterministic answer and evidence metrics for one lane."""
    normalized_answer = normalize_answer(answer)
    normalized_gold = normalize_answer(case.answer)
    recall = evidence_recall(case.expected_document_keys, relevant_documents)
    matched = None
    if case.expected_document_keys:
        observed: set[str] = set()
        for document in relevant_documents:
            observed.update(_document_keys(document))
        matched = len(set(case.expected_document_keys) & observed)
    return {
        "exact_match": normalized_answer == normalized_gold,
        "token_f1": token_f1(answer, case.answer),
        "containment": bool(
            normalized_gold
            and (normalized_gold in normalized_answer or normalized_answer in normalized_gold)
        ),
        "evidence_recall": recall,
        "expected_evidence_count": len(case.expected_document_keys),
        "matched_evidence_count": matched,
    }


def agentic_payload(
    case: BenchmarkCase,
    config: RunConfig,
    *,
    use_wiki_context: bool,
) -> dict[str, Any]:
    """Build one agentic-search request payload."""
    payload: dict[str, Any] = {
        "question": case.question,
        "search_type": config.search_type,
        "search_limit": config.search_limit,
        "max_rewrites": config.max_rewrites,
        "use_wiki_context": use_wiki_context,
    }
    if config.search_filter is not None:
        payload["filter"] = config.search_filter
    if config.min_score is not None:
        payload["min_score"] = config.min_score
    if config.llm_provider is not None:
        payload["llm_provider"] = config.llm_provider
    if config.llm_model is not None:
        payload["llm_model"] = config.llm_model
    if config.llm_temperature is not None:
        payload["llm_temperature"] = config.llm_temperature
    return payload


def assert_lane_payload_parity(
    wiki_off_payload: dict[str, Any],
    wiki_on_payload: dict[str, Any],
) -> None:
    """Assert that use_wiki_context is the only request-payload difference."""
    off = dict(wiki_off_payload)
    on = dict(wiki_on_payload)
    off.pop("use_wiki_context", None)
    on.pop("use_wiki_context", None)
    if off != on:
        raise BenchmarkConfigError("Lane payloads differ beyond use_wiki_context")


def _doc_id(document: dict[str, Any]) -> str | None:
    value = document.get("id")
    return str(value) if value is not None else None


def extract_observability(response: dict[str, Any]) -> dict[str, Any]:
    """Extract structured wiki observability without parsing trace strings."""
    has_structured = any(
        key in response
        for key in (
            "wiki_source_refs",
            "wiki_promotion_status",
            "wiki_promoted_document_ids",
            "retrieved_document_ids",
        )
    )
    if not has_structured:
        return {
            "status": "unavailable",
            "wiki_source_ref_count": None,
            "wiki_promotion_status": None,
            "wiki_promoted_document_ids": [],
            "retrieved_document_ids": [],
        }

    source_refs = response.get("wiki_source_refs")
    promoted_ids = response.get("wiki_promoted_document_ids")
    retrieved_ids = response.get("retrieved_document_ids")
    if not isinstance(promoted_ids, list):
        promoted_docs = response.get("wiki_promoted_documents")
        promoted_ids = [
            doc_id
            for doc_id in (
                _doc_id(doc) for doc in promoted_docs or [] if isinstance(doc, dict)
            )
            if doc_id is not None
        ]
    return {
        "status": "available",
        "wiki_source_ref_count": len(source_refs) if isinstance(source_refs, list) else 0,
        "wiki_promotion_status": response.get("wiki_promotion_status"),
        "wiki_promoted_document_ids": promoted_ids if isinstance(promoted_ids, list) else [],
        "retrieved_document_ids": retrieved_ids if isinstance(retrieved_ids, list) else [],
    }


def _metadata_for_judge(metadata: object) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    allowed = {
        "source",
        "title",
        "url",
        "file_id",
        "doc_id",
        "document_id",
        "multihop_url",
        "multihop_title",
        "multihop_source",
    }
    return {key: value for key, value in metadata.items() if key in allowed}


def _documents_for_judge(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": document.get("id"),
            "snippet": str(document.get("page_content", ""))[:JUDGE_SNIPPET_CHARS],
            "metadata": _metadata_for_judge(document.get("metadata")),
        }
        for document in documents
    ]


def _answer_record_for_judge(label: str, lane: LaneResult) -> dict[str, Any]:
    documents = lane.response.get("relevant_documents")
    if not isinstance(documents, list):
        documents = []
    return {
        "label": label,
        "answer": lane.response.get("generation", ""),
        "raw_evidence_documents": _documents_for_judge(
            [doc for doc in documents if isinstance(doc, dict)]
        ),
    }


def _blinded_lane_entries(
    case: BenchmarkCase,
    wiki_off: LaneResult,
    wiki_on: LaneResult,
    *,
    blind_seed: int,
) -> list[tuple[str, LaneResult]]:
    lanes = sorted(
        [wiki_off, wiki_on],
        key=lambda lane: hashlib.sha256(
            f"{case.id}:{blind_seed}:{lane.lane}".encode()
        ).hexdigest(),
    )
    return list(zip(("A", "B"), lanes, strict=True))


def build_judge_label_to_lane(
    case: BenchmarkCase,
    wiki_off: LaneResult,
    wiki_on: LaneResult,
    *,
    blind_seed: int = DEFAULT_SEED,
) -> dict[str, LaneName]:
    """Build the private mapping needed to interpret blinded judge labels."""
    return {
        label: lane.lane
        for label, lane in _blinded_lane_entries(
            case,
            wiki_off,
            wiki_on,
            blind_seed=blind_seed,
        )
    }


def build_judge_payload(
    case: BenchmarkCase,
    wiki_off: LaneResult,
    wiki_on: LaneResult,
    *,
    blind_seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Build the judge-visible payload, excluding selected wiki metadata."""
    lane_entries = _blinded_lane_entries(
        case,
        wiki_off,
        wiki_on,
        blind_seed=blind_seed,
    )
    return {
        "case_id": case.id,
        "question": case.question,
        "gold_answer": case.answer,
        "answers": [
            _answer_record_for_judge(label, lane)
            for label, lane in lane_entries
        ],
        "rubric": {
            "correctness": "0-5: factual correctness against the gold answer",
            "completeness": "0-5: covers all required answer facets",
            "groundedness": "0-5: supported by raw evidence documents only",
            "preference": "A, B, or tie",
        },
    }


def _judge_prompt(payload: dict[str, Any]) -> str:
    return (
        "Evaluate two blinded RAG answers. Use only the raw_evidence_documents "
        "as grounding evidence. Return one JSON object with keys: correctness, "
        "completeness, groundedness, uses_required_evidence, unsupported_claims, "
        "preference, rationale.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise BenchmarkConfigError("Judge response was not a JSON object")
    return value


def run_llm_judge(
    payload: dict[str, Any],
    *,
    provider: str | None,
    model: str | None,
    temperature: float | None,
) -> dict[str, Any]:
    """Run the configured LangConnect judge model for one case."""
    from langconnect.agent.config import get_agent_llm

    llm = get_agent_llm(provider=provider, model=model, temperature=temperature)
    result = llm.invoke(_judge_prompt(payload))
    content = getattr(result, "content", result)
    return _parse_json_object(str(content))


def _preferred_lane(
    judge_result: dict[str, Any] | None,
    label_to_lane: dict[str, LaneName],
) -> LaneName | Literal["tie"] | None:
    if not isinstance(judge_result, dict):
        return None
    preference = judge_result.get("preference")
    if not isinstance(preference, str):
        return None
    normalized = preference.strip().upper()
    if normalized in {"TIE", "BOTH", "NONE"}:
        return "tie"
    return label_to_lane.get(normalized)


def run_agentic_lane(
    client: httpx.Client,
    *,
    api_base: str,
    case: BenchmarkCase,
    config: RunConfig,
    lane: LaneName,
) -> LaneResult:
    """Execute one REST agentic-search lane."""
    use_wiki_context = lane == "wiki_on"
    payload = agentic_payload(case, config, use_wiki_context=use_wiki_context)
    endpoint = f"{api_base.rstrip('/')}/collections/{config.collection_id}/agentic-search"
    started = time.perf_counter()
    try:
        response = client.post(endpoint, json=payload)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise BenchmarkConfigError("Agentic search response was not an object")
        error = body.get("error")
        return LaneResult(
            lane=lane,
            payload=payload,
            response=body,
            latency_seconds=time.perf_counter() - started,
            error=str(error) if error else None,
        )
    except Exception as exc:
        return LaneResult(
            lane=lane,
            payload=payload,
            response={},
            latency_seconds=time.perf_counter() - started,
            error=str(exc),
        )


def _lane_metrics(case: BenchmarkCase, lane: LaneResult) -> dict[str, Any]:
    documents = lane.response.get("relevant_documents")
    if not isinstance(documents, list):
        documents = []
    document_dicts = [doc for doc in documents if isinstance(doc, dict)]
    return {
        **evaluate_deterministic_metrics(
            str(lane.response.get("generation", "")),
            case,
            document_dicts,
        ),
        "observability": extract_observability(lane.response),
        "latency_seconds": lane.latency_seconds,
        "error": lane.error,
    }


def _lane_to_report(case: BenchmarkCase, lane: LaneResult) -> dict[str, Any]:
    return {
        "lane": lane.lane,
        "payload": lane.payload,
        "response": lane.response,
        "metrics": _lane_metrics(case, lane),
    }


def build_case_report(
    case: BenchmarkCase,
    wiki_off: LaneResult,
    wiki_on: LaneResult,
    *,
    judge_result: dict[str, Any] | None = None,
    judge_label_to_lane: dict[str, LaneName] | None = None,
) -> dict[str, Any]:
    """Build the serializable report entry for one case."""
    judge = None
    if judge_result is not None or judge_label_to_lane is not None:
        label_to_lane = judge_label_to_lane or {}
        judge = {
            "result": judge_result,
            "label_to_lane": label_to_lane,
            "preferred_lane": _preferred_lane(judge_result, label_to_lane),
        }
    return {
        "case": asdict(case),
        "lanes": [
            _lane_to_report(case, wiki_off),
            _lane_to_report(case, wiki_on),
        ],
        "judge": judge,
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _aggregate_lane(case_reports: list[dict[str, Any]], lane: LaneName) -> dict[str, Any]:
    lane_entries = [
        lane_entry
        for case_report in case_reports
        for lane_entry in case_report["lanes"]
        if lane_entry["lane"] == lane
    ]
    f1_values = [entry["metrics"]["token_f1"] for entry in lane_entries]
    recalls = [
        entry["metrics"]["evidence_recall"]
        for entry in lane_entries
        if entry["metrics"]["evidence_recall"] is not None
    ]
    errors = [entry for entry in lane_entries if entry["metrics"]["error"]]
    return {
        "case_count": len(lane_entries),
        "error_count": len(errors),
        "mean_token_f1": _mean(f1_values),
        "mean_evidence_recall": _mean(recalls),
        "exact_match_count": sum(
            1 for entry in lane_entries if entry["metrics"]["exact_match"]
        ),
        "containment_count": sum(
            1 for entry in lane_entries if entry["metrics"]["containment"]
        ),
    }


def _aggregate_judge_preferences(case_reports: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"wiki_off": 0, "wiki_on": 0, "tie": 0, "unparseable": 0}
    for case_report in case_reports:
        judge = case_report.get("judge")
        if not isinstance(judge, dict) or judge.get("result") is None:
            continue
        preferred_lane = judge.get("preferred_lane")
        if preferred_lane in counts:
            counts[preferred_lane] += 1
        else:
            counts["unparseable"] += 1
    return counts


def build_report(
    *,
    cases: list[BenchmarkCase],
    case_reports: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build the full benchmark report."""
    return {
        "metadata": {
            "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "dataset": getattr(args, "dataset_name", DEFAULT_DATASET_NAME),
            "pilot_size": len(cases),
            "seed": args.seed,
            "api_base": args.api_base,
            "collection_id": args.collection_id,
            "search_type": args.search_type,
            "search_limit": args.search_limit,
            "max_rewrites": args.max_rewrites,
            "judge_skipped": args.skip_judge,
        },
        "summary": {
            "wiki_off": _aggregate_lane(case_reports, "wiki_off"),
            "wiki_on": _aggregate_lane(case_reports, "wiki_on"),
            "judge_preferences": _aggregate_judge_preferences(case_reports),
        },
        "cases": case_reports,
    }


def _select_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    cases = load_cases(args.cases)
    if args.case_ids:
        selected = select_cases_by_id(cases, load_case_ids(args.case_ids))
    else:
        selected = select_pilot_cases(
            cases,
            pilot_size=args.pilot_size,
            seed=args.seed,
        )
    if args.limit_cases is not None:
        selected = selected[: args.limit_cases]
    if args.cases_output:
        write_case_ids(selected, args.cases_output)
    return selected


def _format_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['metadata'].get('dataset', DEFAULT_DATASET_NAME)} Wiki Benchmark",
        "",
        "lane | cases | errors | mean_token_f1 | mean_evidence_recall | exact | contains",
        "--- | ---: | ---: | ---: | ---: | ---: | ---:",
    ]
    for lane in ("wiki_off", "wiki_on"):
        summary = report["summary"][lane]
        mean_f1 = summary["mean_token_f1"]
        mean_recall = summary["mean_evidence_recall"]
        lines.append(
            " | ".join(
                [
                    lane,
                    str(summary["case_count"]),
                    str(summary["error_count"]),
                    "" if mean_f1 is None else f"{mean_f1:.4f}",
                    "" if mean_recall is None else f"{mean_recall:.4f}",
                    str(summary["exact_match_count"]),
                    str(summary["containment_count"]),
                ]
            )
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark agentic answers with LLM Wiki off/on.",
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--case-ids", type=Path)
    parser.add_argument("--cases-output", type=Path)
    parser.add_argument("--pilot-size", type=int, default=DEFAULT_PILOT_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--collection-id")
    parser.add_argument(
        "--search-type",
        choices=["semantic", "keyword", "hybrid"],
        default=DEFAULT_SEARCH_TYPE,
    )
    parser.add_argument("--search-limit", type=int, default=DEFAULT_SEARCH_LIMIT)
    parser.add_argument("--max-rewrites", type=int, default=DEFAULT_MAX_REWRITES)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--llm-provider")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-temperature", type=float)
    parser.add_argument("--judge-provider")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-temperature", type=float, default=0)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def run_benchmark(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    """Run benchmark and return report plus exit code."""
    selected_cases = _select_cases(args)
    if args.dry_run:
        report = {
            "metadata": {
                "dataset": args.dataset_name,
                "pilot_size": len(selected_cases),
                "seed": args.seed,
                "dry_run": True,
            },
            "case_ids": [case.id for case in selected_cases],
        }
        return report, 0
    if not args.collection_id:
        raise BenchmarkConfigError("--collection-id is required unless --dry-run")

    config = RunConfig(
        collection_id=args.collection_id,
        search_type=args.search_type,
        search_limit=args.search_limit,
        max_rewrites=args.max_rewrites,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        min_score=args.min_score,
    )
    case_reports: list[dict[str, Any]] = []
    with httpx.Client(timeout=args.timeout) as client:
        for case in selected_cases:
            wiki_off = run_agentic_lane(
                client,
                api_base=args.api_base,
                case=case,
                config=config,
                lane="wiki_off",
            )
            wiki_on = run_agentic_lane(
                client,
                api_base=args.api_base,
                case=case,
                config=config,
                lane="wiki_on",
            )
            assert_lane_payload_parity(wiki_off.payload, wiki_on.payload)
            judge_result = None
            judge_label_to_lane = None
            if not args.skip_judge:
                judge_label_to_lane = build_judge_label_to_lane(
                    case,
                    wiki_off,
                    wiki_on,
                    blind_seed=args.seed,
                )
                judge_payload = build_judge_payload(
                    case,
                    wiki_off,
                    wiki_on,
                    blind_seed=args.seed,
                )
                judge_result = run_llm_judge(
                    judge_payload,
                    provider=args.judge_provider,
                    model=args.judge_model,
                    temperature=args.judge_temperature,
                )
            case_reports.append(
                build_case_report(
                    case,
                        wiki_off,
                        wiki_on,
                        judge_result=judge_result,
                        judge_label_to_lane=judge_label_to_lane,
                    )
                )

    return build_report(cases=selected_cases, case_reports=case_reports, args=args), 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    try:
        report, exit_code = run_benchmark(args)
    except (BenchmarkConfigError, httpx.HTTPError) as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        return 2

    output = (
        _format_markdown(report)
        if args.format == "markdown"
        else json.dumps(report, ensure_ascii=False, indent=2)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
