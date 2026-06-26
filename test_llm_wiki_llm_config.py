# ruff: noqa: S101, SLF001
"""Tests for LLM Wiki-specific LLM environment selection."""

from __future__ import annotations

import pytest

from langconnect.services import llm_wiki


@pytest.mark.asyncio
async def test_get_wiki_llm_uses_sni_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use SNI_LLM_* env values when rebuild args are omitted."""
    calls: dict[str, object] = {}
    fake_llm = object()

    def fake_get_agent_llm(**kwargs: object) -> object:
        calls.update(kwargs)
        return fake_llm

    monkeypatch.setenv(llm_wiki.SNI_LLM_PROVIDER_ENV, "ollama")
    monkeypatch.setenv(
        llm_wiki.SNI_LLM_BASE_URL_ENV,
        "http://host.docker.internal:11434",
    )
    monkeypatch.setenv(llm_wiki.SNI_LLM_MODEL_ENV, "qwen3.5:397b-cloud")
    monkeypatch.setenv(llm_wiki.SNI_LLM_TEMPERATURE_ENV, "0")
    monkeypatch.setattr(llm_wiki, "get_agent_llm", fake_get_agent_llm)

    llm = await llm_wiki._get_wiki_llm(
        llm_provider=None,
        llm_model=None,
        llm_temperature=None,
    )

    assert llm is fake_llm
    assert calls == {
        "provider": "ollama",
        "model": "qwen3.5:397b-cloud",
        "temperature": 0.0,
        "base_url": "http://host.docker.internal:11434",
    }


@pytest.mark.asyncio
async def test_get_wiki_llm_explicit_args_override_wiki_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefer explicit rebuild args over SNI_LLM_* env values."""
    calls: dict[str, object] = {}
    fake_llm = object()

    def fake_get_agent_llm(**kwargs: object) -> object:
        calls.update(kwargs)
        return fake_llm

    monkeypatch.setenv(llm_wiki.SNI_LLM_PROVIDER_ENV, "ollama")
    monkeypatch.setenv(
        llm_wiki.SNI_LLM_BASE_URL_ENV,
        "http://host.docker.internal:11434",
    )
    monkeypatch.setenv(llm_wiki.SNI_LLM_MODEL_ENV, "qwen3.5:397b-cloud")
    monkeypatch.setenv(llm_wiki.SNI_LLM_TEMPERATURE_ENV, "0")
    monkeypatch.setattr(llm_wiki, "get_agent_llm", fake_get_agent_llm)

    llm = await llm_wiki._get_wiki_llm(
        llm_provider="openai",
        llm_model="gpt-5.4",
        llm_temperature=0.2,
    )

    assert llm is fake_llm
    assert calls == {
        "provider": "openai",
        "model": "gpt-5.4",
        "temperature": 0.2,
        "base_url": None,
    }


@pytest.mark.asyncio
async def test_get_wiki_llm_auto_uses_ollama_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use SNI Ollama model in auto mode when the endpoint has it."""
    calls: list[dict[str, object]] = []
    fake_llm = object()

    async def fake_is_ollama_model_available(model: str, base_url: str) -> bool:
        assert model == "qwen3.5:397b-cloud"
        assert base_url == "http://host.docker.internal:11434"
        return True

    def fake_get_agent_llm(**kwargs: object) -> object:
        calls.append(kwargs)
        return fake_llm

    monkeypatch.setenv(llm_wiki.SNI_LLM_PROVIDER_ENV, "auto")
    monkeypatch.setenv(
        llm_wiki.SNI_LLM_BASE_URL_ENV,
        "http://host.docker.internal:11434",
    )
    monkeypatch.setenv(llm_wiki.SNI_LLM_MODEL_ENV, "qwen3.5:397b-cloud")
    monkeypatch.setenv(llm_wiki.SNI_LLM_OPENAI_MODEL_ENV, "gpt-5.4-mini")
    monkeypatch.setenv(llm_wiki.SNI_LLM_TEMPERATURE_ENV, "0")
    monkeypatch.setattr(
        llm_wiki,
        "is_ollama_model_available",
        fake_is_ollama_model_available,
    )
    monkeypatch.setattr(llm_wiki, "get_agent_llm", fake_get_agent_llm)

    llm = await llm_wiki._get_wiki_llm(
        llm_provider=None,
        llm_model=None,
        llm_temperature=None,
    )

    assert llm is fake_llm
    assert calls == [
        {
            "provider": "ollama",
            "model": "qwen3.5:397b-cloud",
            "temperature": 0.0,
            "base_url": "http://host.docker.internal:11434",
        }
    ]


@pytest.mark.asyncio
async def test_get_wiki_llm_auto_uses_openai_when_ollama_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use SNI OpenAI fallback in auto mode when Ollama lacks the model."""
    calls: list[dict[str, object]] = []
    fake_llm = object()

    async def fake_is_ollama_model_available(model: str, base_url: str) -> bool:
        assert model == "qwen3.5:397b-cloud"
        assert base_url == "http://host.docker.internal:11434"
        return False

    def fake_get_agent_llm(**kwargs: object) -> object:
        calls.append(kwargs)
        return fake_llm

    monkeypatch.setenv(llm_wiki.SNI_LLM_PROVIDER_ENV, "auto")
    monkeypatch.setenv(
        llm_wiki.SNI_LLM_BASE_URL_ENV,
        "http://host.docker.internal:11434",
    )
    monkeypatch.setenv(llm_wiki.SNI_LLM_MODEL_ENV, "qwen3.5:397b-cloud")
    monkeypatch.setenv(llm_wiki.SNI_LLM_OPENAI_MODEL_ENV, "gpt-5.4-mini")
    monkeypatch.setenv(llm_wiki.SNI_LLM_TEMPERATURE_ENV, "0")
    monkeypatch.setattr(
        llm_wiki,
        "is_ollama_model_available",
        fake_is_ollama_model_available,
    )
    monkeypatch.setattr(llm_wiki, "get_agent_llm", fake_get_agent_llm)

    llm = await llm_wiki._get_wiki_llm(
        llm_provider=None,
        llm_model=None,
        llm_temperature=None,
    )

    assert llm is fake_llm
    assert calls == [
        {
            "provider": "openai",
            "model": "gpt-5.4-mini",
            "temperature": 0.0,
            "base_url": None,
        }
    ]
