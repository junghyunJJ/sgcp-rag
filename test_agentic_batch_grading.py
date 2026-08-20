# ruff: noqa: S101
"""Regression checks for batched agentic document grading."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langconnect.agent.config import create_chat_model
from langconnect.agent.graders import get_document_grader
from langconnect.agent.nodes import grade_documents

DOCUMENTS = [
    {
        "id": "doc-0",
        "page_content": "PubMedBERT is a biomedical embedding model.",
        "metadata": {},
    },
    {
        "id": "doc-1",
        "page_content": "This chunk discusses an unrelated weather forecast.",
        "metadata": {},
    },
    {
        "id": "wiki-doc",
        "page_content": "Biomedical language models support literature retrieval.",
        "metadata": {"wiki_promoted": True},
    },
]


def test_document_grader_parses_relevant_indices() -> None:
    """Parse one batch JSON response into relevant document indices."""
    llm = FakeListChatModel(
        responses=['{"relevant_indices": [0, 2]}'],
    )

    result = get_document_grader(llm).invoke(
        {
            "question": "Which chunks discuss biomedical models?",
            "documents": "[0]\nfirst\n\n[1]\nsecond\n\n[2]\nthird",
        }
    )

    assert result.relevant_indices == [0, 2]


@pytest.mark.asyncio
async def test_grade_documents_batches_once_and_preserves_order() -> None:
    """Grade normal and Wiki-promoted documents once in source order."""
    grader = MagicMock()
    grader.ainvoke = AsyncMock(
        return_value=MagicMock(relevant_indices=[2, 0, 2]),
    )

    with patch(
        "langconnect.agent.nodes.get_document_grader",
        return_value=grader,
    ):
        result = await grade_documents(
            {
                "question": "Which chunks discuss biomedical models?",
                "documents": DOCUMENTS,
            },
            MagicMock(),
        )

    grader.ainvoke.assert_awaited_once()
    payload = grader.ainvoke.await_args.args[0]
    assert payload == {
        "question": "Which chunks discuss biomedical models?",
        "documents": (
            "[0]\nPubMedBERT is a biomedical embedding model.\n\n"
            "[1]\nThis chunk discusses an unrelated weather forecast.\n\n"
            "[2]\nBiomedical language models support literature retrieval."
        ),
    }
    assert result["relevant_documents"] == [DOCUMENTS[0], DOCUMENTS[2]]
    assert result["steps"] == ["grade_documents: 2/3 relevant"]


@pytest.mark.asyncio
async def test_grade_documents_skips_llm_for_empty_input() -> None:
    """Return no relevant documents without constructing a grader."""
    with patch("langconnect.agent.nodes.get_document_grader") as factory:
        result = await grade_documents(
            {"question": "anything", "documents": []},
            MagicMock(),
        )

    factory.assert_not_called()
    assert result == {
        "relevant_documents": [],
        "steps": ["grade_documents: 0/0 relevant"],
    }


@pytest.mark.asyncio
async def test_grade_documents_rejects_out_of_range_index() -> None:
    """Reject a model response that references a missing document."""
    grader = MagicMock()
    grader.ainvoke = AsyncMock(
        return_value=MagicMock(relevant_indices=[len(DOCUMENTS)]),
    )

    with (
        patch(
            "langconnect.agent.nodes.get_document_grader",
            return_value=grader,
        ),
        pytest.raises(ValueError, match="out-of-range document index"),
    ):
        await grade_documents(
            {"question": "anything", "documents": DOCUMENTS},
            MagicMock(),
        )


def test_ollama_agent_model_disables_reasoning() -> None:
    """Keep thinking disabled for the local agent model."""
    llm = create_chat_model(
        provider="ollama",
        model="qwen3.5:122b",
        temperature=0,
        base_url="http://localhost:4000",
    )

    assert llm.reasoning is False
