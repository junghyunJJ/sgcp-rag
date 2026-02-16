"""Tests for Agentic RAG search.

All LLM calls and Collection.search() are mocked —
no database or API keys required.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mark all async tests in this module
pytestmark = pytest.mark.asyncio

from langconnect.agent.state import AgentState
from langconnect.agent.graders import (
    GradeAnswer,
    GradeDocumentRelevance,
    GradeHallucination,
)
from langconnect.agent.nodes import (
    generate,
    grade_documents,
    grade_generation,
    retrieve,
    rewrite_query,
)


# --- Fixtures ---

MOCK_DOCUMENTS = [
    {
        "id": "doc-1",
        "page_content": "LangGraph is a framework for building stateful agents.",
        "metadata": {"source": "test.pdf"},
        "score": 0.95,
    },
    {
        "id": "doc-2",
        "page_content": "Python is a programming language.",
        "metadata": {"source": "test2.pdf"},
        "score": 0.60,
    },
]


def _make_state(**overrides) -> AgentState:
    """Create a base AgentState with sensible defaults."""
    base = {
        "question": "What is LangGraph?",
        "collection_id": "test-collection-uuid",
        "user_id": None,
        "search_type": "hybrid",
        "search_limit": 5,
        "search_filter": None,
        "documents": [],
        "relevant_documents": [],
        "generation": "",
        "query_rewrites": [],
        "rewrite_count": 0,
        "max_rewrites": 3,
        "steps": [],
        "error": None,
    }
    base.update(overrides)
    return base


def _mock_llm_with_structured_output(binary_score: str):
    """Create a mock LLM that returns structured output with given score."""
    mock_result = MagicMock()
    mock_result.binary_score = binary_score

    mock_chain = AsyncMock(return_value=mock_result)

    mock_structured = MagicMock()
    mock_structured.__or__ = MagicMock(return_value=mock_chain)

    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)
    return mock_llm, mock_chain


def _mock_llm_for_generation(answer_text: str):
    """Create a mock LLM for answer generation."""
    mock_result = MagicMock()
    mock_result.content = answer_text

    mock_chain = AsyncMock(return_value=mock_result)

    mock_llm = MagicMock()
    mock_llm.__or__ = MagicMock(return_value=mock_chain)
    return mock_llm, mock_chain


# --- State Tests ---


def test_agent_state_defaults():
    """AgentState should accept all expected fields."""
    state = _make_state()
    assert state["question"] == "What is LangGraph?"
    assert state["rewrite_count"] == 0
    assert state["max_rewrites"] == 3
    assert state["steps"] == []


# --- Node Tests ---


@patch("langconnect.agent.nodes.Collection")
async def test_retrieve_node(mock_collection_class):
    """Retrieve node should call Collection.search() and return documents."""
    mock_instance = AsyncMock()
    mock_instance.search = AsyncMock(return_value=MOCK_DOCUMENTS)
    mock_collection_class.return_value = mock_instance

    state = _make_state()
    result = await retrieve(state)

    assert len(result["documents"]) == 2
    assert "retrieve:" in result["steps"][0]
    mock_instance.search.assert_awaited_once_with(
        "What is LangGraph?",
        limit=5,
        search_type="hybrid",
        filter=None,
    )


async def test_grade_documents_all_relevant():
    """All documents graded as relevant should be kept."""
    mock_grader = AsyncMock()
    mock_grader.ainvoke = AsyncMock(
        return_value=MagicMock(binary_score="yes")
    )

    state = _make_state(documents=MOCK_DOCUMENTS)

    with patch("langconnect.agent.nodes.get_document_grader", return_value=mock_grader):
        mock_llm = MagicMock()
        result = await grade_documents(state, mock_llm)

    assert len(result["relevant_documents"]) == 2
    assert "2/2 relevant" in result["steps"][-1]


async def test_grade_documents_partial_relevant():
    """Only relevant documents should be kept."""
    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        score = "yes" if call_count == 1 else "no"
        return MagicMock(binary_score=score)

    mock_grader = AsyncMock()
    mock_grader.ainvoke = AsyncMock(side_effect=side_effect)

    state = _make_state(documents=MOCK_DOCUMENTS)

    with patch("langconnect.agent.nodes.get_document_grader", return_value=mock_grader):
        mock_llm = MagicMock()
        result = await grade_documents(state, mock_llm)

    assert len(result["relevant_documents"]) == 1
    assert "1/2 relevant" in result["steps"][-1]


async def test_generate_node():
    """Generate node should produce an answer from relevant docs."""
    mock_result = MagicMock()
    mock_result.content = "LangGraph is a framework for building stateful agents."

    # chain.ainvoke() needs to return mock_result
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_result)

    state = _make_state(relevant_documents=MOCK_DOCUMENTS)

    with patch("langconnect.agent.nodes.ChatPromptTemplate") as mock_prompt_cls:
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_prompt_cls.from_messages = MagicMock(return_value=mock_prompt)

        mock_llm = MagicMock()
        result = await generate(state, mock_llm)

    assert result["generation"] == "LangGraph is a framework for building stateful agents."
    assert "generate:" in result["steps"][-1]


async def test_rewrite_query_node():
    """Rewrite node should produce a new question and increment counter."""
    mock_result = MagicMock()
    mock_result.content = "How does LangGraph work for building agents?"

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_result)

    state = _make_state(rewrite_count=1, query_rewrites=["previous rewrite"])

    with patch("langconnect.agent.nodes.ChatPromptTemplate") as mock_prompt_cls:
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_prompt_cls.from_messages = MagicMock(return_value=mock_prompt)

        mock_llm = MagicMock()
        result = await rewrite_query(state, mock_llm)

    assert result["question"] == "How does LangGraph work for building agents?"
    assert result["rewrite_count"] == 2
    assert len(result["query_rewrites"]) == 2
    assert "rewrite_query:" in result["steps"][-1]


async def test_grade_generation_passes():
    """Grade generation should pass when both checks succeed."""
    mock_hallucination_grader = AsyncMock()
    mock_hallucination_grader.ainvoke = AsyncMock(
        return_value=MagicMock(binary_score="yes")
    )
    mock_answer_grader = AsyncMock()
    mock_answer_grader.ainvoke = AsyncMock(
        return_value=MagicMock(binary_score="yes")
    )

    state = _make_state(
        generation="LangGraph builds stateful agents.",
        relevant_documents=MOCK_DOCUMENTS,
    )

    with (
        patch("langconnect.agent.nodes.get_hallucination_grader", return_value=mock_hallucination_grader),
        patch("langconnect.agent.nodes.get_answer_grader", return_value=mock_answer_grader),
    ):
        mock_llm = MagicMock()
        result = await grade_generation(state, mock_llm)

    assert "PASSED" in result["steps"][-1]


async def test_grade_generation_fails_hallucination():
    """Grade generation should fail on hallucination check."""
    mock_hallucination_grader = AsyncMock()
    mock_hallucination_grader.ainvoke = AsyncMock(
        return_value=MagicMock(binary_score="no")
    )

    state = _make_state(
        generation="LangGraph was invented by OpenAI.",
        relevant_documents=MOCK_DOCUMENTS,
    )

    with patch("langconnect.agent.nodes.get_hallucination_grader", return_value=mock_hallucination_grader):
        mock_llm = MagicMock()
        result = await grade_generation(state, mock_llm)

    assert "FAILED hallucination" in result["steps"][-1]


async def test_grade_generation_fails_answer_quality():
    """Grade generation should fail on answer quality check."""
    mock_hallucination_grader = AsyncMock()
    mock_hallucination_grader.ainvoke = AsyncMock(
        return_value=MagicMock(binary_score="yes")
    )
    mock_answer_grader = AsyncMock()
    mock_answer_grader.ainvoke = AsyncMock(
        return_value=MagicMock(binary_score="no")
    )

    state = _make_state(
        generation="I don't know.",
        relevant_documents=MOCK_DOCUMENTS,
    )

    with (
        patch("langconnect.agent.nodes.get_hallucination_grader", return_value=mock_hallucination_grader),
        patch("langconnect.agent.nodes.get_answer_grader", return_value=mock_answer_grader),
    ):
        mock_llm = MagicMock()
        result = await grade_generation(state, mock_llm)

    assert "FAILED answer quality" in result["steps"][-1]


# --- Graph Routing Tests ---


def test_route_after_grading_with_relevant_docs():
    """Should route to generate when relevant docs exist."""
    from langconnect.agent.graph import _route_after_grading

    state = _make_state(relevant_documents=MOCK_DOCUMENTS)
    assert _route_after_grading(state) == "generate"


def test_route_after_grading_no_relevant_docs():
    """Should route to rewrite when no relevant docs and rewrites available."""
    from langconnect.agent.graph import _route_after_grading

    state = _make_state(relevant_documents=[], rewrite_count=0, max_rewrites=3)
    assert _route_after_grading(state) == "rewrite_query"


def test_route_after_grading_max_rewrites_reached():
    """Should force generate when max rewrites reached even without relevant docs."""
    from langconnect.agent.graph import _route_after_grading

    state = _make_state(relevant_documents=[], rewrite_count=3, max_rewrites=3)
    assert _route_after_grading(state) == "generate"


def test_route_after_generation_check_passed():
    """Should route to END when generation passes."""
    from langconnect.agent.graph import _route_after_generation_check
    from langgraph.graph import END

    state = _make_state(steps=["grade_generation: PASSED both checks"])
    assert _route_after_generation_check(state) == END


def test_route_after_generation_check_failed_with_retries():
    """Should route to rewrite when generation fails and retries available."""
    from langconnect.agent.graph import _route_after_generation_check

    state = _make_state(
        steps=["grade_generation: FAILED hallucination check"],
        rewrite_count=1,
        max_rewrites=3,
    )
    assert _route_after_generation_check(state) == "rewrite_query"


def test_route_after_generation_check_failed_max_rewrites():
    """Should route to END when generation fails but no retries left."""
    from langconnect.agent.graph import _route_after_generation_check
    from langgraph.graph import END

    state = _make_state(
        steps=["grade_generation: FAILED answer quality check"],
        rewrite_count=3,
        max_rewrites=3,
    )
    assert _route_after_generation_check(state) == END


# --- Entry Point Tests ---


async def test_run_agentic_search_error_handling():
    """run_agentic_search should catch exceptions and return error dict."""
    from langconnect.agent import run_agentic_search

    with patch(
        "langconnect.agent.build_agentic_rag_graph",
        side_effect=RuntimeError("LLM unavailable"),
    ):
        result = await run_agentic_search(
            question="test?",
            collection_id="fake-uuid",
        )

    assert result["error"] is not None
    assert "LLM unavailable" in result["error"]
    assert result["generation"] == ""


# --- E2E Graph Test ---


async def test_full_graph_e2e():
    """Run the compiled graph end-to-end with mocked LLM and Collection."""
    from langconnect.agent.graph import build_agentic_rag_graph

    # Mock graders — all return "yes"
    mock_doc_grader = AsyncMock()
    mock_doc_grader.ainvoke = AsyncMock(
        return_value=MagicMock(binary_score="yes")
    )
    mock_hallucination_grader = AsyncMock()
    mock_hallucination_grader.ainvoke = AsyncMock(
        return_value=MagicMock(binary_score="yes")
    )
    mock_answer_grader = AsyncMock()
    mock_answer_grader.ainvoke = AsyncMock(
        return_value=MagicMock(binary_score="yes")
    )

    # Mock LLM for generate/rewrite (prompt | llm → chain)
    mock_answer = MagicMock()
    mock_answer.content = "LangGraph is a framework for stateful agents."

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_answer)

    mock_llm = MagicMock()
    # prompt | llm returns mock_chain (used by generate and rewrite_query)
    mock_llm.__or__ = MagicMock(return_value=mock_chain)

    # Patch Collection.search and all grader factories
    with (
        patch("langconnect.agent.nodes.Collection") as mock_coll_cls,
        patch("langconnect.agent.nodes.get_document_grader", return_value=mock_doc_grader),
        patch("langconnect.agent.nodes.get_hallucination_grader", return_value=mock_hallucination_grader),
        patch("langconnect.agent.nodes.get_answer_grader", return_value=mock_answer_grader),
        patch("langconnect.agent.nodes.ChatPromptTemplate") as mock_prompt_cls,
    ):
        # Collection.search returns MOCK_DOCUMENTS
        mock_instance = AsyncMock()
        mock_instance.search = AsyncMock(return_value=MOCK_DOCUMENTS)
        mock_coll_cls.return_value = mock_instance

        # ChatPromptTemplate.from_messages | llm → mock_chain
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_prompt_cls.from_messages = MagicMock(return_value=mock_prompt)

        graph = build_agentic_rag_graph(mock_llm)

        initial_state = _make_state()
        result = await graph.ainvoke(initial_state)

    # Verify the graph produced a generation and passed all checks
    assert result["generation"] == "LangGraph is a framework for stateful agents."
    assert any("PASSED" in step for step in result["steps"])
    assert result["rewrite_count"] == 0  # No rewrites needed
    assert len(result["relevant_documents"]) == 2


async def test_full_graph_e2e_with_rewrite():
    """E2E test where first retrieval fails grading, triggering a rewrite loop."""
    from langconnect.agent.graph import build_agentic_rag_graph

    # Track call count to simulate: first retrieval → no relevant docs,
    # second retrieval (after rewrite) → relevant docs
    doc_grade_call = {"count": 0}

    async def doc_grader_side_effect(*args, **kwargs):
        doc_grade_call["count"] += 1
        # First 2 calls (first retrieval): both irrelevant
        # Next 2 calls (after rewrite): both relevant
        if doc_grade_call["count"] <= 2:
            return MagicMock(binary_score="no")
        return MagicMock(binary_score="yes")

    mock_doc_grader = AsyncMock()
    mock_doc_grader.ainvoke = AsyncMock(side_effect=doc_grader_side_effect)

    mock_hallucination_grader = AsyncMock()
    mock_hallucination_grader.ainvoke = AsyncMock(
        return_value=MagicMock(binary_score="yes")
    )
    mock_answer_grader = AsyncMock()
    mock_answer_grader.ainvoke = AsyncMock(
        return_value=MagicMock(binary_score="yes")
    )

    mock_answer = MagicMock()
    mock_answer.content = "Rewritten answer about LangGraph."

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_answer)

    mock_llm = MagicMock()
    mock_llm.__or__ = MagicMock(return_value=mock_chain)

    with (
        patch("langconnect.agent.nodes.Collection") as mock_coll_cls,
        patch("langconnect.agent.nodes.get_document_grader", return_value=mock_doc_grader),
        patch("langconnect.agent.nodes.get_hallucination_grader", return_value=mock_hallucination_grader),
        patch("langconnect.agent.nodes.get_answer_grader", return_value=mock_answer_grader),
        patch("langconnect.agent.nodes.ChatPromptTemplate") as mock_prompt_cls,
    ):
        mock_instance = AsyncMock()
        mock_instance.search = AsyncMock(return_value=MOCK_DOCUMENTS)
        mock_coll_cls.return_value = mock_instance

        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_prompt_cls.from_messages = MagicMock(return_value=mock_prompt)

        graph = build_agentic_rag_graph(mock_llm)
        initial_state = _make_state(max_rewrites=3)
        result = await graph.ainvoke(initial_state)

    # Should have rewritten once, then succeeded
    assert result["rewrite_count"] == 1
    assert len(result["query_rewrites"]) == 1
    assert result["generation"] == "Rewritten answer about LangGraph."
    assert any("PASSED" in step for step in result["steps"])
