"""Query expansion LLM configuration and execution."""

import os

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate

from langconnect.agent.config import (
    create_chat_model,
    get_query_expansion_ollama_base_url,
    is_ollama_model_available,
)

DEFAULT_QUERY_EXPANSION_PROVIDER = "auto"
DEFAULT_QUERY_EXPANSION_OLLAMA_MODEL = "qwen3.5:35b"
DEFAULT_QUERY_EXPANSION_OPENAI_MODEL = "gpt-5.4"
QUERY_EXPANSION_PROMPT = """You are an AI language model assistant. Your task is to generate 3 to 5
different versions of the given user question to retrieve relevant documents from a vector
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions separated by newlines. Do not number them.
Original question: {question}"""


class QueryExpansionError(RuntimeError):
    """Raised when query expansion cannot be configured or executed."""


class QueryExpansionProviderUnavailableError(QueryExpansionError):
    """Raised when a fallback-eligible provider is unavailable."""


class LineListOutputParser(BaseOutputParser[list[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> list[str]:
        """Parse newline-delimited model output into clean query strings."""
        lines = [line.strip() for line in text.strip().split("\n")]
        return [line for line in lines if line]


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _query_expansion_temperature() -> float:
    return _env_float("QUERY_EXPANSION_LLM_TEMPERATURE", "0")


def _query_expansion_provider() -> str:
    return os.getenv(
        "QUERY_EXPANSION_LLM_PROVIDER",
        DEFAULT_QUERY_EXPANSION_PROVIDER,
    ).strip().lower()


def _query_expansion_model() -> str:
    return os.getenv(
        "QUERY_EXPANSION_LLM_MODEL",
        DEFAULT_QUERY_EXPANSION_OLLAMA_MODEL,
    )


def _query_expansion_openai_model() -> str:
    if model := os.getenv("QUERY_EXPANSION_OPENAI_MODEL"):
        return model

    if os.getenv("AGENT_LLM_PROVIDER", "openai").strip().lower() == "openai":
        return os.getenv("AGENT_LLM_MODEL", DEFAULT_QUERY_EXPANSION_OPENAI_MODEL)

    return DEFAULT_QUERY_EXPANSION_OPENAI_MODEL


def _query_expansion_openai_temperature() -> float:
    return _env_float(
        "QUERY_EXPANSION_OPENAI_TEMPERATURE",
        str(_query_expansion_temperature()),
    )


def _url_is_under_base(url: object, base_url: str) -> bool:
    current = str(url).rstrip("/")
    expected = base_url.rstrip("/")
    return current == expected or current.startswith(f"{expected}/")


def _safe_getattr(value: object, name: str) -> object | None:
    try:
        return getattr(value, name, None)
    except RuntimeError:
        return None


def _httpx_error_targets_base(error: BaseException, base_url: str) -> bool:
    for source in (_safe_getattr(error, "request"), _safe_getattr(error, "response")):
        request = _safe_getattr(source, "request") or source
        url = _safe_getattr(request, "url")
        if url is not None and _url_is_under_base(url, base_url):
            return True
    return False


def _is_query_expansion_fallback_eligible(
    error: BaseException,
    base_url: str,
) -> bool:
    if type(error).__module__.startswith("ollama"):
        return True

    if isinstance(error, httpx.HTTPError) and _httpx_error_targets_base(
        error,
        base_url,
    ):
        return True

    message = str(error).lower()
    return "ollama" in message and any(
        marker in message
        for marker in ("connect", "failed", "refused", "timeout", "unavailable")
    )


async def _invoke_query_expansion(llm: BaseChatModel, question: str) -> list[str]:
    query_prompt = PromptTemplate(
        input_variables=["question"],
        template=QUERY_EXPANSION_PROMPT,
    )
    chain = query_prompt | llm | LineListOutputParser()
    return await chain.ainvoke({"question": question})


async def _generate_with_ollama(question: str) -> list[str]:
    model = _query_expansion_model()
    temperature = _query_expansion_temperature()
    base_url = get_query_expansion_ollama_base_url()

    if not await is_ollama_model_available(model, base_url):
        raise QueryExpansionProviderUnavailableError(
            f"Ollama model {model!r} is not available at {base_url!r}",
        )

    try:
        llm = create_chat_model(
            provider="ollama",
            model=model,
            temperature=temperature,
            base_url=base_url,
        )
        return await _invoke_query_expansion(llm, question)
    except QueryExpansionError:
        raise
    except Exception as exc:
        message = f"Ollama query expansion failed: {exc}"
        if _is_query_expansion_fallback_eligible(exc, base_url):
            raise QueryExpansionProviderUnavailableError(message) from exc
        raise QueryExpansionError(message) from exc


async def _generate_with_openai(question: str) -> list[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise QueryExpansionError("OpenAI API key not configured")

    try:
        llm = create_chat_model(
            provider="openai",
            model=_query_expansion_openai_model(),
            temperature=_query_expansion_openai_temperature(),
            base_url=None,
            api_key=api_key,
        )
        return await _invoke_query_expansion(llm, question)
    except QueryExpansionError:
        raise
    except Exception as exc:
        raise QueryExpansionError(f"OpenAI query expansion failed: {exc}") from exc


async def generate_query_expansions(question: str) -> list[str]:
    """Generate search query variants using configured query expansion policy."""
    provider = _query_expansion_provider()

    if provider == "ollama":
        return await _generate_with_ollama(question)

    if provider == "openai":
        return await _generate_with_openai(question)

    if provider != "auto":
        raise QueryExpansionError(
            "Unsupported query expansion provider: "
            f"{provider!r}. Use 'auto', 'ollama', or 'openai'.",
        )

    try:
        return await _generate_with_ollama(question)
    except QueryExpansionProviderUnavailableError as ollama_error:
        try:
            return await _generate_with_openai(question)
        except QueryExpansionError as openai_error:
            raise QueryExpansionError(
                f"{ollama_error}; OpenAI fallback failed: {openai_error}",
            ) from openai_error
