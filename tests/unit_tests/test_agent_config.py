"""Tests for agent LLM configuration."""

import os
from unittest.mock import patch

import pytest

from langconnect.agent.config import get_agent_llm


def test_default_openai_provider():
    """Default provider should be OpenAI."""
    llm = get_agent_llm()
    from langchain_openai import ChatOpenAI

    assert isinstance(llm, ChatOpenAI)


def test_openai_provider_explicit():
    """Explicit OpenAI provider."""
    llm = get_agent_llm(provider="openai", model="gpt-4.1-nano", temperature=0.5)
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
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_agent_llm(provider="anthropic")


def test_env_var_defaults():
    """Environment variables should set defaults."""
    with patch.dict(os.environ, {
        "AGENT_LLM_PROVIDER": "openai",
        "AGENT_LLM_MODEL": "gpt-4.1-mini",
        "AGENT_LLM_TEMPERATURE": "0.7",
    }):
        llm = get_agent_llm()
        from langchain_openai import ChatOpenAI

        assert isinstance(llm, ChatOpenAI)


def test_parameter_overrides_env():
    """Explicit parameters should override env vars."""
    with patch.dict(os.environ, {
        "AGENT_LLM_PROVIDER": "google",
        "AGENT_LLM_MODEL": "gemini-pro",
    }):
        # Explicit openai should override google env var
        llm = get_agent_llm(provider="openai", model="gpt-4.1-nano")
        from langchain_openai import ChatOpenAI

        assert isinstance(llm, ChatOpenAI)
