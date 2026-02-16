"""Agentic RAG LLM configuration.

Supports OpenAI and Google providers with environment-based defaults
and per-request overrides via API parameters.
"""

import os

from langchain_core.language_models import BaseChatModel


def get_agent_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    """Create an LLM instance for the agent.

    Args:
        provider: "openai" or "google". Defaults to AGENT_LLM_PROVIDER env var.
        model: Model name. Defaults to AGENT_LLM_MODEL env var.
        temperature: Sampling temperature. Defaults to AGENT_LLM_TEMPERATURE env var.

    Returns:
        A LangChain chat model instance.

    Raises:
        ValueError: If the provider is not supported.
    """
    provider = provider or os.getenv("AGENT_LLM_PROVIDER", "openai")
    model = model or os.getenv("AGENT_LLM_MODEL", "gpt-4.1-nano")
    temperature = (
        temperature
        if temperature is not None
        else float(os.getenv("AGENT_LLM_TEMPERATURE", "0"))
    )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    raise ValueError(f"Unsupported LLM provider: {provider!r}. Use 'openai' or 'google'.")
