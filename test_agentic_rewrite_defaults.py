# ruff: noqa: PLR2004, S101
"""Regression checks for the default agentic rewrite budget."""

from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest

import langconnect.agent as agent_module
from langconnect.agent.graph import _route_after_grading
from langconnect.models.agentic import AgenticSearchQuery
from mcpserver import mcp_server, mcp_sse_server


def test_public_agentic_defaults_use_two_rewrites() -> None:
    """Expose a two-rewrite default through REST and both MCP transports."""
    assert AgenticSearchQuery(question="question").max_rewrites == 2
    assert (
        inspect.signature(mcp_server.agentic_search.fn)
        .parameters["max_rewrites"]
        .default
        == 2
    )
    assert (
        inspect.signature(mcp_sse_server.agentic_search.fn)
        .parameters["max_rewrites"]
        .default
        == 2
    )


@pytest.mark.asyncio
async def test_direct_agentic_default_uses_two_rewrites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use two rewrites when a direct caller and environment omit the limit."""
    monkeypatch.delenv("AGENT_MAX_REWRITES", raising=False)
    captured_state: dict[str, object] = {}

    async def fake_invoke(**kwargs: object) -> dict[str, object]:
        captured_state.update(kwargs["initial_state"])  # type: ignore[arg-type]
        return dict(captured_state)

    with patch.object(agent_module, "_invoke_agent_graph", side_effect=fake_invoke):
        await agent_module.run_agentic_search(
            question="question",
            collection_id="collection-id",
            llm_provider="openai",
            use_wiki_context=False,
        )

    assert captured_state["max_rewrites"] == 2


def test_graph_fallback_uses_two_rewrites() -> None:
    """Stop no-context retrieval when an incomplete state reaches two rewrites."""
    assert (
        _route_after_grading(
            {
                "relevant_documents": [],
                "rewrite_count": 2,
            }
        )
        == "no_context"
    )
