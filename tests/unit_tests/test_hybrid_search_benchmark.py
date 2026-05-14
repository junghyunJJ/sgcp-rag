import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "benchmark_hybrid_search.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark_hybrid_search", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
sys.modules["benchmark_hybrid_search"] = bench
SPEC.loader.exec_module(bench)


def _result(
    result_id: str,
    source: str,
    score: float,
    content: str = "content",
) -> dict[str, object]:
    return {
        "id": result_id,
        "score": score,
        "page_content": content,
        "metadata": {"source": source},
    }


def test_resolve_collection_by_unique_name() -> None:
    """A unique collection name should resolve to its UUID."""
    collections = [
        {
            "uuid": "collection-1",
            "name": "agentpaper",
            "document_count": 27,
            "chunk_count": 2974,
        },
    ]

    assert bench.resolve_collection_id(collections, "agentpaper") == "collection-1"


def test_resolve_collection_requires_id_for_duplicate_names() -> None:
    """Duplicate collection names should fail deterministically."""
    collections = [
        {"uuid": "collection-1", "name": "agentpaper"},
        {"uuid": "collection-2", "name": "agentpaper"},
    ]

    with pytest.raises(bench.BenchmarkConfigError) as exc_info:
        bench.resolve_collection_id(collections, "agentpaper")

    message = str(exc_info.value)
    assert "Multiple collections named 'agentpaper'" in message
    assert "collection-1" in message
    assert "collection-2" in message


def test_score_expected_top_source() -> None:
    """Top-source expectations should pass only when rank 1 matches."""
    case = bench.BenchmarkCase(
        name="agent skills",
        query="agent skill",
        expectation="top_source",
        expected_source="skillfoundry.pdf",
    )
    results = [
        _result("a", "skillfoundry.pdf", 0.8),
        _result("b", "SkillClaw.pdf", 0.7),
    ]

    evaluation = bench.evaluate_results(case, "current_hybrid", results)

    assert evaluation.passed is True
    assert evaluation.expected_rank == 1
    assert evaluation.top_source == "skillfoundry.pdf"


def test_score_expected_source_in_top_n_warns_on_late_rank() -> None:
    """Top-N expectations should pass while preserving rank warnings."""
    case = bench.BenchmarkCase(
        name="voyager",
        query="Voyager executable skills",
        expectation="source_in_top_n",
        expected_source="skillfoundry.pdf",
        top_n=3,
        warning_if_expected_rank_gt=1,
    )
    results = [
        _result("a", "SkillClaw.pdf", 0.48),
        _result("b", "Other.pdf", 0.4),
        _result("c", "skillfoundry.pdf", 0.3),
    ]

    evaluation = bench.evaluate_results(case, "current_hybrid", results)

    assert evaluation.passed is True
    assert evaluation.warning is True
    assert evaluation.expected_rank == 3


def test_score_expected_empty() -> None:
    """Empty expectations should pass only when no rows are returned."""
    case = bench.BenchmarkCase(
        name="noise",
        query="zzzz qwerty nonexistent agentpaper noise",
        expectation="empty",
        expected_empty=True,
    )

    assert bench.evaluate_results(case, "current_hybrid", []).passed is True
    assert (
        bench.evaluate_results(
            case,
            "current_hybrid",
            [_result("a", "SkillClaw.pdf", 0.2)],
        ).passed
        is False
    )


def test_score_filter_all_match() -> None:
    """Filter expectations should verify every returned row source."""
    case = bench.BenchmarkCase(
        name="filter skillfoundry",
        query="agent skill",
        expectation="filter_all_match",
        expected_source="skillfoundry.pdf",
        filter={"source": "skillfoundry.pdf"},
    )

    passed = bench.evaluate_results(
        case,
        "current_hybrid",
        [
            _result("a", "skillfoundry.pdf", 0.8),
            _result("b", "skillfoundry.pdf", 0.7),
        ],
    )
    failed = bench.evaluate_results(
        case,
        "current_hybrid",
        [
            _result("a", "skillfoundry.pdf", 0.8),
            _result("b", "SkillClaw.pdf", 0.7),
        ],
    )

    assert passed.passed is True
    assert failed.passed is False


def test_fuse_results_dedupes_and_applies_candidate_weights() -> None:
    """Offline fusion should dedupe by string id and apply candidate weights."""
    semantic = [
        _result("42", "semantic.pdf", 0.8, "semantic content"),
        _result("semantic-only", "semantic.pdf", 0.5, "semantic only"),
    ]
    keyword = [
        _result(42, "keyword.pdf", 1.0, "keyword content"),
        _result("keyword-only", "keyword.pdf", 0.5, "keyword only"),
    ]

    fused = bench.fuse_results(semantic, keyword, semantic_weight=0.6)

    assert [row["id"] for row in fused] == ["42", "semantic-only", "keyword-only"]
    assert fused[0]["score"] == pytest.approx(0.88)
    assert fused[0]["page_content"] == "semantic content"
    assert fused[1]["score"] == pytest.approx(0.3)
    assert fused[2]["score"] == pytest.approx(0.2)


def test_exit_code_is_exploratory_by_default() -> None:
    """Expectation failures should only fail the process when requested."""
    evaluations = [
        bench.CaseEvaluation(
            case_name="voyager",
            lane="current_hybrid",
            passed=False,
            warning=False,
            result_count=1,
            expected_rank=None,
            top_source="SkillClaw.pdf",
            top_score=0.48,
            top_sources=["SkillClaw.pdf"],
        )
    ]

    assert bench.exit_code_for(evaluations, fail_on_regression=False) == 0
    assert bench.exit_code_for(evaluations, fail_on_regression=True) == 1


def test_exit_code_gates_only_current_hybrid_regressions() -> None:
    """Exploratory lane failures should not make regression gating fail."""
    evaluations = [
        bench.CaseEvaluation(
            case_name="noise",
            lane="offline_fusion_0.7_0.3",
            passed=False,
            warning=False,
            result_count=8,
            expected_rank=None,
            top_source="SkillClaw.pdf",
            top_score=0.44,
            top_sources=["SkillClaw.pdf"],
        ),
        bench.CaseEvaluation(
            case_name="noise",
            lane="current_hybrid",
            passed=True,
            warning=False,
            result_count=0,
            expected_rank=None,
            top_source=None,
            top_score=None,
            top_sources=[],
        ),
    ]

    assert bench.exit_code_for(evaluations, fail_on_regression=True) == 0
