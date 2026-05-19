"""Tests for agent LLM configuration."""

import os
from unittest.mock import patch

import pytest

from langconnect.agent.config import get_agent_llm


def test_default_openai_provider():
    """Default provider should be OpenAI."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
        llm = get_agent_llm()
    from langchain_openai import ChatOpenAI

    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-5.4"


def test_openai_provider_explicit():
    """Explicit OpenAI provider."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
        llm = get_agent_llm(
            provider="openai",
            model="gpt-4.1-nano",
            temperature=0.5,
        )
    from langchain_openai import ChatOpenAI

    assert isinstance(llm, ChatOpenAI)


def test_google_provider():
    """Google provider should create ChatGoogleGenerativeAI."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
        llm = get_agent_llm(provider="google", model="gemini-2.0-flash")
        from langchain_google_genai import ChatGoogleGenerativeAI

        assert isinstance(llm, ChatGoogleGenerativeAI)


def test_unsupported_provider():
    """Unsupported provider should raise ValueError."""
    with pytest.raises(ValueError, match="'openai', 'google', or 'ollama'"):
        get_agent_llm(provider="anthropic")


def test_env_var_defaults():
    """Environment variables should set defaults."""
    with patch.dict(os.environ, {
        "AGENT_LLM_PROVIDER": "openai",
        "AGENT_LLM_MODEL": "gpt-4.1-mini",
        "AGENT_LLM_TEMPERATURE": "0.7",
        "OPENAI_API_KEY": "test_key",
    }):
        llm = get_agent_llm()
        from langchain_openai import ChatOpenAI

        assert isinstance(llm, ChatOpenAI)


def test_parameter_overrides_env():
    """Explicit parameters should override env vars."""
    with patch.dict(os.environ, {
        "AGENT_LLM_PROVIDER": "google",
        "AGENT_LLM_MODEL": "gemini-pro",
        "OPENAI_API_KEY": "test_key",
    }):
        # Explicit openai should override google env var
        llm = get_agent_llm(provider="openai", model="gpt-4.1-nano")
        from langchain_openai import ChatOpenAI

        assert isinstance(llm, ChatOpenAI)


def test_ollama_provider_uses_env_base_url():
    """Ollama provider should use ChatOllama with the configured base URL."""
    with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:5000"}):
        llm = get_agent_llm(
            provider="ollama",
            model="qwen3.5:122b",
            temperature=0,
        )

    from langchain_ollama import ChatOllama

    assert isinstance(llm, ChatOllama)
    assert llm.model == "qwen3.5:122b"
    assert llm.base_url == "http://localhost:5000"
    assert llm.reasoning is False


def test_ollama_provider_uses_agent_base_url_over_global():
    """Agentic RAG Ollama should prefer its dedicated endpoint over the global one."""
    with patch.dict(
        os.environ,
        {
            "AGENT_OLLAMA_BASE_URL": "http://localhost:6200",
            "OLLAMA_BASE_URL": "http://localhost:5000",
        },
    ):
        llm = get_agent_llm(
            provider="ollama",
            model="qwen3.5:122b",
            temperature=0,
        )

    from langchain_ollama import ChatOllama

    assert isinstance(llm, ChatOllama)
    assert llm.model == "qwen3.5:122b"
    assert llm.base_url == "http://localhost:6200"
    assert llm.reasoning is False


def test_ollama_provider_defaults_to_agentic_rag_model():
    """Ollama agentic RAG should default to the large configured local model."""
    with patch.dict(os.environ, {"AGENT_LLM_PROVIDER": "ollama"}, clear=True):
        llm = get_agent_llm()

    from langchain_ollama import ChatOllama

    assert isinstance(llm, ChatOllama)
    assert llm.model == "qwen3.5:122b"
    assert llm.reasoning is False
