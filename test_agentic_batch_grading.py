# ruff: noqa: PLR2004, S101, SLF001
"""Regression checks for batched agentic document grading."""

from __future__ import annotations

from types import SimpleNamespace
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

LARGE_DOCUMENTS = [
    {
        "id": f"large-{index}",
        "page_content": str(index) * 7_000,
        "metadata": {"wiki_promoted": index == 3},
    }
    for index in range(4)
]

TOKEN_DENSE_DOCUMENTS = [
    {
        "id": f"dense-{index}",
        "page_content": "한" * 900,
        "metadata": {},
    }
    for index in range(4)
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


@pytest.mark.asyncio
async def test_grade_documents_splits_large_input_and_merges_global_indices() -> None:
    """Split large prompts while preserving global indices and source order."""
    grader = MagicMock()
    llm = MagicMock()
    llm.num_ctx = 25_100
    grader.ainvoke = AsyncMock(
        side_effect=[
            MagicMock(relevant_indices=[2, 0]),
            MagicMock(relevant_indices=[3]),
        ]
    )

    with patch(
        "langconnect.agent.nodes.get_document_grader",
        return_value=grader,
    ):
        result = await grade_documents(
            {"question": "question", "documents": LARGE_DOCUMENTS},
            llm,
        )

    assert grader.ainvoke.await_count == 2
    first_payload = grader.ainvoke.await_args_list[0].args[0]["documents"]
    second_payload = grader.ainvoke.await_args_list[1].args[0]["documents"]
    assert first_payload.startswith("[0]\n")
    assert "\n\n[1]\n" in first_payload
    assert "\n\n[2]\n" in first_payload
    assert "[3]\n" not in first_payload
    assert second_payload.startswith("[3]\n")
    assert result["relevant_documents"] == [
        LARGE_DOCUMENTS[0],
        LARGE_DOCUMENTS[2],
        LARGE_DOCUMENTS[3],
    ]
    assert result["steps"] == ["grade_documents: 3/4 relevant"]


@pytest.mark.asyncio
async def test_grade_documents_respects_context_for_token_dense_text() -> None:
    """Keep token-dense document batches inside a 4K model context."""
    grader = MagicMock()
    llm = MagicMock()
    llm.num_ctx = 4_096
    grader.ainvoke = AsyncMock(
        side_effect=[
            MagicMock(relevant_indices=[index])
            for index in range(len(TOKEN_DENSE_DOCUMENTS))
        ]
    )

    with patch(
        "langconnect.agent.nodes.get_document_grader",
        return_value=grader,
    ):
        result = await grade_documents(
            {"question": "question", "documents": TOKEN_DENSE_DOCUMENTS},
            llm,
        )

    assert grader.ainvoke.await_count == len(TOKEN_DENSE_DOCUMENTS)
    assert result["relevant_documents"] == TOKEN_DENSE_DOCUMENTS


@pytest.mark.asyncio
async def test_grade_documents_uses_running_ollama_context() -> None:
    """Use Ollama's active context when num_ctx is not set explicitly."""
    grader = MagicMock()
    grader.ainvoke = AsyncMock(
        side_effect=[
            MagicMock(relevant_indices=[index])
            for index in range(len(TOKEN_DENSE_DOCUMENTS))
        ]
    )
    llm = create_chat_model(
        provider="ollama",
        model="qwen3.5:122b",
        temperature=0,
        base_url="http://localhost:4000",
    )
    running_models = SimpleNamespace(
        models=[
            SimpleNamespace(
                model="qwen3.5:122b",
                context_length=4_096,
            )
        ]
    )

    with (
        patch(
            "langconnect.agent.nodes.get_document_grader",
            return_value=grader,
        ),
        patch.object(
            llm._async_client,
            "ps",
            AsyncMock(return_value=running_models),
        ) as ps,
    ):
        await grade_documents(
            {"question": "question", "documents": TOKEN_DENSE_DOCUMENTS},
            llm,
        )

    ps.assert_awaited_once()
    assert grader.ainvoke.await_count == len(TOKEN_DENSE_DOCUMENTS)


@pytest.mark.asyncio
async def test_grade_documents_rejects_index_from_another_batch() -> None:
    """Reject a valid global index absent from the current prompt."""
    grader = MagicMock()
    llm = MagicMock()
    llm.num_ctx = 25_100
    grader.ainvoke = AsyncMock(
        side_effect=[MagicMock(relevant_indices=[3])],
    )

    with (
        patch(
            "langconnect.agent.nodes.get_document_grader",
            return_value=grader,
        ),
        pytest.raises(ValueError, match="outside its grading batch"),
    ):
        await grade_documents(
            {"question": "question", "documents": LARGE_DOCUMENTS},
            llm,
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
