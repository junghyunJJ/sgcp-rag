"""Tests for Agentic RAG search.

All LLM calls and Collection.search() are mocked —
no database or API keys required.
"""

import operator
from typing import Annotated, get_args, get_origin, get_type_hints
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError

from langconnect.agent.nodes import (
    generate,
    grade_documents,
    grade_generation,
    retrieve,
    rewrite_query,
)
from langconnect.agent.state import AgentState

# Mark all async tests in this module
pytestmark = pytest.mark.asyncio


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


def _make_state(**overrides: object) -> AgentState:
    """Create a base AgentState with sensible defaults."""
    base = {
        "question": "What is LangGraph?",
        "collection_id": "test-collection-uuid",
        "search_type": "hybrid",
        "search_limit": 5,
        "search_filter": None,
        "min_score": None,
        "documents": [],
        "relevant_documents": [],
        "generation": "",
        "query_rewrites": [],
        "rewrite_count": 0,
        "max_rewrites": 3,
        "steps": [],
        "error": None,
        "no_context_found": False,
        "use_wiki_context": False,
        "wiki_context": "",
        "selected_wiki_pages": [],
        "wiki_context_status": "disabled",
    }
    base.update(overrides)
    return base


def _mock_llm_with_structured_output(
    binary_score: str,
) -> tuple[MagicMock, AsyncMock]:
    """Create a mock LLM that returns structured output with given score."""
    mock_result = MagicMock()
    mock_result.binary_score = binary_score

    mock_chain = AsyncMock(return_value=mock_result)

    mock_structured = MagicMock()
    mock_structured.__or__ = MagicMock(return_value=mock_chain)

    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)
    return mock_llm, mock_chain


def _mock_llm_for_generation(answer_text: str) -> tuple[MagicMock, AsyncMock]:
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
    assert state["min_score"] is None
    assert state["no_context_found"] is False
    assert state["use_wiki_context"] is False
    assert state["wiki_context"] == ""
    assert state["selected_wiki_pages"] == []
    assert state["wiki_context_status"] == "disabled"


@pytest.mark.parametrize("field_name", ["steps", "query_rewrites"])
async def test_agent_state_uses_reducers_for_accumulated_lists(field_name: str):
    """LangGraph should accumulate trace lists without each node copying state."""
    hints = get_type_hints(AgentState, include_extras=True)

    assert get_origin(hints[field_name]) is Annotated
    assert operator.add in get_args(hints[field_name])


def test_agentic_search_query_accepts_bounded_min_score():
    """AgenticSearchQuery should accept min_score in the 0..1 range."""
    from langconnect.models.agentic import AgenticSearchQuery

    assert AgenticSearchQuery(question="test", min_score=0).min_score == 0
    assert AgenticSearchQuery(question="test", min_score=0.75).min_score == 0.75
    assert AgenticSearchQuery(question="test", min_score=1).min_score == 1
    assert AgenticSearchQuery(question="test").use_wiki_context is False
    assert AgenticSearchQuery(question="test", use_wiki_context=True).use_wiki_context


def test_agentic_search_result_defaults_wiki_metadata():
    """AgenticSearchResult should default wiki metadata to disabled/empty."""
    from langconnect.models.agentic import AgenticSearchResult

    result = AgenticSearchResult()

    assert result.selected_wiki_pages == []
    assert result.wiki_context_status == "disabled"


@pytest.mark.parametrize("min_score", [-0.1, 1.1])
def test_agentic_search_query_rejects_out_of_range_min_score(min_score):
    """AgenticSearchQuery should bound min_score to 0..1."""
    from langconnect.models.agentic import AgenticSearchQuery

    with pytest.raises(ValidationError):
        AgenticSearchQuery(question="test", min_score=min_score)


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
        min_score=None,
    )


@patch("langconnect.agent.nodes.Collection")
async def test_retrieve_node_forwards_min_score(mock_collection_class):
    """Retrieve node should pass min_score to Collection.search()."""
    mock_instance = AsyncMock()
    mock_instance.search = AsyncMock(return_value=MOCK_DOCUMENTS)
    mock_collection_class.return_value = mock_instance

    state = _make_state(min_score=0.82)
    await retrieve(state)

    mock_instance.search.assert_awaited_once_with(
        "What is LangGraph?",
        limit=5,
        search_type="hybrid",
        filter=None,
        min_score=0.82,
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

    async def side_effect(*args: object, **kwargs: object) -> MagicMock:
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
    mock_chain.ainvoke.assert_awaited_once_with(
        {
            "question": "What is LangGraph?",
            "context": (
                "LangGraph is a framework for building stateful agents."
                "\n\n---\n\n"
                "Python is a programming language."
            ),
        }
    )


async def test_generate_node_with_wiki_context_uses_wiki_prompt():
    """Generate should include wiki context only when it is selected."""
    mock_result = MagicMock()
    mock_result.content = "LangGraph is a framework for building stateful agents."

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_result)

    state = _make_state(
        relevant_documents=MOCK_DOCUMENTS,
        use_wiki_context=True,
        wiki_context="Non-authoritative navigation memory: LangGraph agents.",
        selected_wiki_pages=[{"id": "lg", "title": "LangGraph"}],
        wiki_context_status="selected",
    )

    with patch("langconnect.agent.nodes.ChatPromptTemplate") as mock_prompt_cls:
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_prompt_cls.from_messages = MagicMock(return_value=mock_prompt)

        mock_llm = MagicMock()
        result = await generate(state, mock_llm)

    prompt_text = mock_prompt_cls.from_messages.call_args.args[0][0][1]
    assert "non-authoritative navigation memory" in prompt_text
    mock_chain.ainvoke.assert_awaited_once_with(
        {
            "question": "What is LangGraph?",
            "context": (
                "LangGraph is a framework for building stateful agents."
                "\n\n---\n\n"
                "Python is a programming language."
            ),
            "wiki_context": "Non-authoritative navigation memory: LangGraph agents.",
        }
    )
    assert result["generation"] == "LangGraph is a framework for building stateful agents."
    assert "wiki_context: selected 1 pages" in result["steps"]


async def test_no_context_node_sets_terminal_error():
    """No-context terminal node should clear generated content and documents."""
    from langconnect.agent.nodes import no_context

    state = _make_state(
        generation="stale answer",
        relevant_documents=MOCK_DOCUMENTS,
        steps=["grade_documents: 0/2 relevant"],
    )

    result = await no_context(state)

    assert result["error"] == "no_relevant_context"
    assert result["generation"] == ""
    assert result["relevant_documents"] == []
    assert result["no_context_found"] is True
    assert result["steps"][-1] == "no_context: no relevant documents found"


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
    assert result["query_rewrites"] == [
        "How does LangGraph work for building agents?"
    ]
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
        wiki_context="Non-authoritative navigation memory that must be ignored.",
    )

    with (
        patch("langconnect.agent.nodes.get_hallucination_grader", return_value=mock_hallucination_grader),
        patch("langconnect.agent.nodes.get_answer_grader", return_value=mock_answer_grader),
    ):
        mock_llm = MagicMock()
        result = await grade_generation(state, mock_llm)

    assert "PASSED" in result["steps"][-1]
    hallucination_payload = mock_hallucination_grader.ainvoke.await_args.args[0]
    assert "Non-authoritative navigation memory" not in hallucination_payload["documents"]


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
    """Should route to no_context when max rewrites reached without relevant docs."""
    from langconnect.agent.graph import _route_after_grading

    state = _make_state(relevant_documents=[], rewrite_count=3, max_rewrites=3)
    assert _route_after_grading(state) == "no_context"


def test_route_after_generation_check_passed():
    """Should route to END when generation passes."""
    from langgraph.graph import END

    from langconnect.agent.graph import _route_after_generation_check

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
    from langgraph.graph import END

    from langconnect.agent.graph import _route_after_generation_check

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
    ), patch("langconnect.agent.get_agent_llm", return_value=MagicMock()):
        result = await run_agentic_search(
            question="test?",
            collection_id="fake-uuid",
        )

    assert result["error"] is not None
    assert "LLM unavailable" in result["error"]
    assert result["generation"] == ""
    assert result["selected_wiki_pages"] == []
    assert result["wiki_context_status"] == "disabled"


async def test_run_agentic_search_passes_min_score_to_initial_state():
    """run_agentic_search should propagate min_score into graph state."""
    from langconnect.agent import run_agentic_search

    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "generation": "answer",
            "relevant_documents": [],
            "steps": [],
            "query_rewrites": [],
            "rewrite_count": 0,
            "error": None,
        }
    )

    with (
        patch("langconnect.agent.get_agent_llm", return_value=MagicMock()),
        patch("langconnect.agent.build_agentic_rag_graph", return_value=mock_graph),
    ):
        await run_agentic_search(
            question="test?",
            collection_id="fake-uuid",
            min_score=0.73,
        )

    initial_state = mock_graph.ainvoke.await_args.args[0]
    assert initial_state["min_score"] == 0.73
    assert initial_state["use_wiki_context"] is False
    assert initial_state["wiki_context_status"] == "disabled"


async def test_run_agentic_search_resolves_wiki_context_when_enabled():
    """run_agentic_search should load selected wiki context into graph state."""
    from langconnect.agent import run_agentic_search
    from langconnect.agent.wiki_context import WikiContextResult

    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "generation": "answer",
            "relevant_documents": [],
            "steps": [],
            "query_rewrites": [],
            "rewrite_count": 0,
            "error": None,
            "selected_wiki_pages": [{"id": "wiki", "title": "Wiki"}],
            "wiki_context_status": "selected",
        }
    )

    with (
        patch("langconnect.agent.get_agent_llm", return_value=MagicMock()),
        patch("langconnect.agent.build_agentic_rag_graph", return_value=mock_graph),
        patch(
            "langconnect.agent.resolve_wiki_context",
            return_value=WikiContextResult(
                context="Non-authoritative navigation memory.",
                selected_pages=[{"id": "wiki", "title": "Wiki"}],
                status="selected",
            ),
        ) as mock_resolve,
    ):
        result = await run_agentic_search(
            question="test?",
            collection_id="fake-uuid",
            use_wiki_context=True,
        )

    mock_resolve.assert_called_once_with("fake-uuid", "test?")
    initial_state = mock_graph.ainvoke.await_args.args[0]
    assert initial_state["use_wiki_context"] is True
    assert initial_state["wiki_context"] == "Non-authoritative navigation memory."
    assert initial_state["selected_wiki_pages"] == [{"id": "wiki", "title": "Wiki"}]
    assert initial_state["wiki_context_status"] == "selected"
    assert result["selected_wiki_pages"] == [{"id": "wiki", "title": "Wiki"}]
    assert result["wiki_context_status"] == "selected"


async def test_run_agentic_search_preserves_finite_wiki_status_on_error():
    """Graph setup failures should not erase resolved wiki status metadata."""
    from langconnect.agent import run_agentic_search
    from langconnect.agent.wiki_context import WikiContextResult

    with (
        patch(
            "langconnect.agent.resolve_wiki_context",
            return_value=WikiContextResult(
                context="Non-authoritative navigation memory.",
                selected_pages=[{"id": "wiki", "title": "Wiki"}],
                status="selected",
            ),
        ),
        patch("langconnect.agent.get_agent_llm", return_value=MagicMock()),
        patch(
            "langconnect.agent.build_agentic_rag_graph",
            side_effect=RuntimeError("graph unavailable"),
        ),
    ):
        result = await run_agentic_search(
            question="test?",
            collection_id="fake-uuid",
            use_wiki_context=True,
        )

    assert result["error"] == "graph unavailable"
    assert result["selected_wiki_pages"] == [{"id": "wiki", "title": "Wiki"}]
    assert result["wiki_context_status"] == "selected"


async def test_agentic_api_passes_min_score_to_runner():
    """API endpoint should forward request min_score to run_agentic_search()."""
    from langconnect.api.agentic import agentic_search
    from langconnect.models.agentic import AgenticSearchQuery

    with patch(
        "langconnect.api.agentic.run_agentic_search",
        new=AsyncMock(
            return_value={
                "generation": "",
                "relevant_documents": [],
                "steps": [],
                "query_rewrites": [],
                "rewrite_count": 0,
                "error": None,
            }
        ),
    ) as mock_runner:
        await agentic_search(
            uuid4(),
            AgenticSearchQuery(
                question="What is LangGraph?",
                min_score=0.64,
                use_wiki_context=True,
            ),
        )

    assert mock_runner.await_args.kwargs["min_score"] == 0.64
    assert mock_runner.await_args.kwargs["use_wiki_context"] is True


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

    async def doc_grader_side_effect(
        *args: object,
        **kwargs: object,
    ) -> MagicMock:
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


async def test_full_graph_no_context_does_not_generate():
    """Max rewrites with no relevant docs should terminate without generation."""
    from langconnect.agent.graph import build_agentic_rag_graph

    mock_doc_grader = AsyncMock()
    mock_doc_grader.ainvoke = AsyncMock(return_value=MagicMock(binary_score="no"))

    mock_llm = MagicMock()

    with (
        patch("langconnect.agent.nodes.Collection") as mock_coll_cls,
        patch("langconnect.agent.nodes.get_document_grader", return_value=mock_doc_grader),
        patch("langconnect.agent.graph.generate", new_callable=AsyncMock) as mock_generate,
    ):
        mock_instance = AsyncMock()
        mock_instance.search = AsyncMock(return_value=[])
        mock_coll_cls.return_value = mock_instance

        graph = build_agentic_rag_graph(mock_llm)
        result = await graph.ainvoke(_make_state(max_rewrites=0))

    mock_generate.assert_not_awaited()
    assert result["error"] == "no_relevant_context"
    assert result["generation"] == ""
    assert result["relevant_documents"] == []
    assert result["no_context_found"] is True
