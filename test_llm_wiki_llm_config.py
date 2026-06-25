"""Tests for LLM Wiki-specific LLM environment selection."""

from __future__ import annotations

# ruff: noqa: S101, SLF001

import pytest

from langconnect.services import llm_wiki


def test_get_wiki_llm_uses_wiki_env(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}
    fake_llm = object()

    def fake_get_agent_llm(**kwargs: object) -> object:
        calls.update(kwargs)
        return fake_llm

    monkeypatch.setenv(llm_wiki.WIKI_LLM_PROVIDER_ENV, "ollama")
    monkeypatch.setenv(
        llm_wiki.WIKI_LLM_BASE_URL_ENV,
        "http://host.docker.internal:7000",
    )
    monkeypatch.setenv(llm_wiki.WIKI_LLM_MODEL_ENV, "qwen3.5:35b")
    monkeypatch.setenv(llm_wiki.WIKI_LLM_TEMPERATURE_ENV, "0")
    monkeypatch.setattr(llm_wiki, "get_agent_llm", fake_get_agent_llm)

    llm = llm_wiki._get_wiki_llm(
        llm_provider=None,
        llm_model=None,
        llm_temperature=None,
    )

    assert llm is fake_llm
    assert calls == {
        "provider": "ollama",
        "model": "qwen3.5:35b",
        "temperature": 0.0,
        "base_url": "http://host.docker.internal:7000",
    }


def test_get_wiki_llm_explicit_args_override_wiki_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    fake_llm = object()

    def fake_get_agent_llm(**kwargs: object) -> object:
        calls.update(kwargs)
        return fake_llm

    monkeypatch.setenv(llm_wiki.WIKI_LLM_PROVIDER_ENV, "ollama")
    monkeypatch.setenv(
        llm_wiki.WIKI_LLM_BASE_URL_ENV,
        "http://host.docker.internal:7000",
    )
    monkeypatch.setenv(llm_wiki.WIKI_LLM_MODEL_ENV, "qwen3.5:35b")
    monkeypatch.setenv(llm_wiki.WIKI_LLM_TEMPERATURE_ENV, "0")
    monkeypatch.setattr(llm_wiki, "get_agent_llm", fake_get_agent_llm)

    llm = llm_wiki._get_wiki_llm(
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
