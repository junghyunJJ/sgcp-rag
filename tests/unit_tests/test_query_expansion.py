"""Tests for query expansion LLM configuration."""

import os
from typing import Never
from unittest.mock import patch

import pytest

from langconnect.agent import query_expansion as qe


@pytest.mark.asyncio
async def test_auto_query_expansion_uses_available_ollama(monkeypatch):
    """Auto mode should prefer the configured Ollama query expansion model."""
    calls: list[tuple] = []
    fake_llm = object()

    async def fake_available(model: str, base_url: str) -> bool:
        calls.append(("available", model, base_url))
        return True

    def fake_create_chat_model(**kwargs: object) -> object:
        calls.append(("create", kwargs))
        return fake_llm

    async def fake_invoke(llm, question: str) -> list[str]:
        calls.append(("invoke", llm, question))
        return ["query one", "query two"]

    monkeypatch.setattr(qe, "is_ollama_model_available", fake_available)
    monkeypatch.setattr(qe, "create_chat_model", fake_create_chat_model)
    monkeypatch.setattr(qe, "_invoke_query_expansion", fake_invoke)

    with patch.dict(
        os.environ,
        {
            "QUERY_EXPANSION_LLM_PROVIDER": "auto",
            "QUERY_EXPANSION_LLM_MODEL": "qwen3.5:35b",
            "QUERY_EXPANSION_LLM_TEMPERATURE": "0",
            "QUERY_EXPANSION_OLLAMA_BASE_URL": "http://localhost:5000",
            "OLLAMA_BASE_URL": "http://localhost:5000",
            "OPENAI_API_KEY": "test-key",
        },
    ):
        result = await qe.generate_query_expansions("ask?")

    assert result == ["query one", "query two"]
    assert ("available", "qwen3.5:35b", "http://localhost:5000") in calls
    assert (
        "create",
        {
            "provider": "ollama",
            "model": "qwen3.5:35b",
            "temperature": 0.0,
            "base_url": "http://localhost:5000",
        },
    ) in calls


@pytest.mark.asyncio
async def test_query_expansion_uses_dedicated_ollama_base_url(monkeypatch):
    """Query expansion should prefer its dedicated Ollama endpoint."""
    calls: list[tuple] = []
    fake_llm = object()

    async def fake_available(model: str, base_url: str) -> bool:
        calls.append(("available", model, base_url))
        return True

    def fake_create_chat_model(**kwargs: object) -> object:
        calls.append(("create", kwargs))
        return fake_llm

    async def fake_invoke(llm: object, question: str) -> list[str]:
        calls.append(("invoke", llm, question))
        return ["query one"]

    monkeypatch.setattr(qe, "is_ollama_model_available", fake_available)
    monkeypatch.setattr(qe, "create_chat_model", fake_create_chat_model)
    monkeypatch.setattr(qe, "_invoke_query_expansion", fake_invoke)

    with patch.dict(
        os.environ,
        {
            "QUERY_EXPANSION_LLM_PROVIDER": "auto",
            "QUERY_EXPANSION_LLM_MODEL": "qwen3.5:35b",
            "QUERY_EXPANSION_OLLAMA_BASE_URL": "http://localhost:5100",
            "OLLAMA_BASE_URL": "http://localhost:5000",
            "OPENAI_API_KEY": "test-key",
        },
    ):
        result = await qe.generate_query_expansions("ask?")

    assert result == ["query one"]
    assert ("available", "qwen3.5:35b", "http://localhost:5100") in calls
    assert (
        "create",
        {
            "provider": "ollama",
            "model": "qwen3.5:35b",
            "temperature": 0.0,
            "base_url": "http://localhost:5100",
        },
    ) in calls


@pytest.mark.asyncio
async def test_auto_query_expansion_falls_back_to_openai(monkeypatch):
    """Auto mode should use OpenAI when the configured Ollama model is unavailable."""
    calls: list[tuple] = []
    fake_llm = object()

    async def fake_available(model: str, base_url: str) -> bool:
        calls.append(("available", model, base_url))
        return False

    def fake_create_chat_model(**kwargs: object) -> object:
        calls.append(("create", kwargs))
        return fake_llm

    async def fake_invoke(llm, question: str) -> list[str]:
        calls.append(("invoke", llm, question))
        return ["fallback query"]

    monkeypatch.setattr(qe, "is_ollama_model_available", fake_available)
    monkeypatch.setattr(qe, "create_chat_model", fake_create_chat_model)
    monkeypatch.setattr(qe, "_invoke_query_expansion", fake_invoke)

    with patch.dict(
        os.environ,
        {
            "QUERY_EXPANSION_LLM_PROVIDER": "auto",
            "QUERY_EXPANSION_LLM_MODEL": "qwen3.5:35b",
            "QUERY_EXPANSION_OPENAI_MODEL": "gpt-5.4",
            "OPENAI_API_KEY": "test-key",
        },
    ):
        result = await qe.generate_query_expansions("ask?")

    assert result == ["fallback query"]
    assert (
        "create",
        {
            "provider": "openai",
            "model": "gpt-5.4",
            "temperature": 0.0,
            "base_url": None,
            "api_key": "test-key",
        },
    ) in calls


@pytest.mark.asyncio
async def test_auto_query_expansion_does_not_fallback_for_parser_errors(monkeypatch):
    """Auto mode should not hide non-provider Ollama invocation failures."""
    calls: list[tuple] = []
    fake_llm = object()

    async def fake_available(model: str, base_url: str) -> bool:
        calls.append(("available", model, base_url))
        return True

    def fake_create_chat_model(**kwargs: object) -> object:
        calls.append(("create", kwargs))
        return fake_llm

    async def fake_invoke(llm: object, question: str) -> list[str]:
        calls.append(("invoke", llm, question))
        raise ValueError("parser bug")

    monkeypatch.setattr(qe, "is_ollama_model_available", fake_available)
    monkeypatch.setattr(qe, "create_chat_model", fake_create_chat_model)
    monkeypatch.setattr(qe, "_invoke_query_expansion", fake_invoke)

    with (
        patch.dict(
            os.environ,
            {
                "QUERY_EXPANSION_LLM_PROVIDER": "auto",
                "QUERY_EXPANSION_LLM_MODEL": "qwen3.5:35b",
                "QUERY_EXPANSION_OLLAMA_BASE_URL": "http://localhost:5000",
                "QUERY_EXPANSION_OPENAI_MODEL": "gpt-5.4",
                "OPENAI_API_KEY": "test-key",
            },
        ),
        pytest.raises(qe.QueryExpansionError, match="Ollama query expansion failed"),
    ):
        await qe.generate_query_expansions("ask?")

    created_providers = [item[1]["provider"] for item in calls if item[0] == "create"]
    assert created_providers == ["ollama"]


@pytest.mark.asyncio
async def test_explicit_ollama_query_expansion_does_not_fallback(monkeypatch):
    """Explicit Ollama mode should surface availability errors instead of falling back."""

    async def fake_available(model: str, base_url: str) -> bool:
        return False

    def fail_if_created(**kwargs: object) -> Never:
        raise AssertionError("create_chat_model should not be called")

    monkeypatch.setattr(qe, "is_ollama_model_available", fake_available)
    monkeypatch.setattr(qe, "create_chat_model", fail_if_created)

    with (
        patch.dict(
            os.environ,
            {
                "QUERY_EXPANSION_LLM_PROVIDER": "ollama",
                "QUERY_EXPANSION_LLM_MODEL": "qwen3.5:35b",
                "OPENAI_API_KEY": "test-key",
            },
        ),
        pytest.raises(qe.QueryExpansionError, match="Ollama"),
    ):
        await qe.generate_query_expansions("ask?")


@pytest.mark.asyncio
async def test_openai_query_expansion_requires_api_key(monkeypatch):
    """Explicit OpenAI mode should fail clearly when credentials are missing."""

    def fail_if_created(**kwargs: object) -> Never:
        raise AssertionError("create_chat_model should not be called")

    monkeypatch.setattr(qe, "create_chat_model", fail_if_created)

    with (
        patch.dict(
            os.environ,
            {"QUERY_EXPANSION_LLM_PROVIDER": "openai"},
            clear=True,
        ),
        pytest.raises(qe.QueryExpansionError, match="OpenAI API key"),
    ):
        await qe.generate_query_expansions("ask?")
