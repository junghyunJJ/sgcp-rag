"""Agentic RAG LLM configuration.

Supports OpenAI, Google, and Ollama providers with environment-based defaults
and per-request overrides via API parameters.
"""

import os
from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel

DEFAULT_AGENT_OPENAI_MODEL = "gpt-5.4"
DEFAULT_AGENT_OLLAMA_MODEL = "qwen3.5:122b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:5000"
GLOBAL_OLLAMA_BASE_URL_ENV = "OLLAMA_BASE_URL"
AGENT_OLLAMA_BASE_URL_ENV = "AGENT_OLLAMA_BASE_URL"
QUERY_EXPANSION_OLLAMA_BASE_URL_ENV = "QUERY_EXPANSION_OLLAMA_BASE_URL"
SUPPORTED_LLM_PROVIDERS = ("openai", "google", "ollama")


def _normalize_provider(provider: str) -> str:
    return provider.strip().lower()


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def get_ollama_base_url(
    base_url: str | None = None,
    *,
    env_var: str | None = None,
) -> str:
    """Return the configured Ollama base URL."""
    if base_url:
        return base_url
    if env_var and (scoped_url := os.getenv(env_var)):
        return scoped_url
    return os.getenv(GLOBAL_OLLAMA_BASE_URL_ENV, DEFAULT_OLLAMA_BASE_URL)


def get_agent_ollama_base_url(base_url: str | None = None) -> str:
    """Return the Agentic RAG Ollama base URL."""
    return get_ollama_base_url(base_url, env_var=AGENT_OLLAMA_BASE_URL_ENV)


def get_query_expansion_ollama_base_url(base_url: str | None = None) -> str:
    """Return the query expansion Ollama base URL."""
    return get_ollama_base_url(
        base_url,
        env_var=QUERY_EXPANSION_OLLAMA_BASE_URL_ENV,
    )


def _api_tags_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/api/tags"


def _extract_ollama_model_names(data: dict[str, Any]) -> set[str]:
    return {
        model["name"]
        for model in data.get("models", [])
        if isinstance(model, dict) and isinstance(model.get("name"), str)
    }


async def is_ollama_model_available(model: str, base_url: str) -> bool:
    """Return whether an Ollama endpoint is reachable and has the selected model."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(_api_tags_url(base_url))
            response.raise_for_status()
            model_names = _extract_ollama_model_names(response.json())
    except Exception:
        return False

    return model in model_names


def create_chat_model(
    *,
    provider: str,
    model: str,
    temperature: float,
    base_url: str | None = None,
    api_key: str | None = None,
) -> BaseChatModel:
    """Create a LangChain chat model for the requested provider."""
    provider = _normalize_provider(provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = {"model": model, "temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        return ChatOpenAI(**kwargs)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=get_ollama_base_url(base_url),
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider!r}. Use 'openai', 'google', or 'ollama'.",
    )


def get_agent_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
) -> BaseChatModel:
    """Create an LLM instance for the agent.

    Args:
        provider: "openai", "google", or "ollama".
            Defaults to AGENT_LLM_PROVIDER env var.
        model: Model name. Defaults to AGENT_LLM_MODEL env var.
        temperature: Sampling temperature. Defaults to AGENT_LLM_TEMPERATURE env var.
        base_url: Optional Ollama endpoint override.

    Returns:
        A LangChain chat model instance.

    Raises:
        ValueError: If the provider is not supported.
    """
    provider = _normalize_provider(
        provider or os.getenv("AGENT_LLM_PROVIDER", "openai")
    )
    default_model = (
        DEFAULT_AGENT_OLLAMA_MODEL
        if provider == "ollama"
        else DEFAULT_AGENT_OPENAI_MODEL
    )
    model = model or os.getenv("AGENT_LLM_MODEL", default_model)
    temperature = (
        temperature
        if temperature is not None
        else _env_float("AGENT_LLM_TEMPERATURE", "0")
    )

    return create_chat_model(
        provider=provider,
        model=model,
        temperature=temperature,
        base_url=get_agent_ollama_base_url(base_url) if provider == "ollama" else None,
    )
