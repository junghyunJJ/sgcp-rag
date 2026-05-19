"""Agentic RAG grader models.

Pydantic models used to normalize binary yes/no decisions for document relevance,
hallucination detection, and answer quality.
"""

import json
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langconnect.agent.prompts import (
    ANSWER_GRADER_PROMPT,
    DOCUMENT_GRADER_PROMPT,
    HALLUCINATION_GRADER_PROMPT,
)


class GradeDocumentRelevance(BaseModel):
    """Binary score for document relevance to a question."""

    binary_score: Literal["yes", "no"] = Field(
        description="Document relevance: 'yes' or 'no'"
    )


class GradeHallucination(BaseModel):
    """Binary score for whether generation is grounded in facts."""

    binary_score: Literal["yes", "no"] = Field(
        description="Answer grounded in facts: 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary score for whether answer addresses the question."""

    binary_score: Literal["yes", "no"] = Field(
        description="Answer addresses question: 'yes' or 'no'"
    )


def _normalize_binary_score(value: object) -> Literal["yes", "no"]:
    if not isinstance(value, str):
        raise ValueError("binary_score must be 'yes' or 'no'")

    score = value.strip().lower().rstrip(".!")
    if score in {"yes", "no"}:
        return score  # type: ignore[return-value]

    raise ValueError("binary_score must be 'yes' or 'no'")


def parse_binary_score_response(
    text: str,
    model_class: type[BaseModel],
) -> BaseModel:
    """Parse schema-shaped JSON or a bare yes/no response into a grader model."""
    stripped = text.strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        return model_class(
            binary_score=_normalize_binary_score(parsed.get("binary_score"))
        )

    if isinstance(parsed, str):
        return model_class(binary_score=_normalize_binary_score(parsed))

    return model_class(binary_score=_normalize_binary_score(stripped))


class BinaryScoreOutputParser(BaseOutputParser[BaseModel]):
    """Parse binary grader output into the requested Pydantic model."""

    model_class: type[BaseModel]

    def parse(self, text: str) -> BaseModel:
        """Parse text into the configured binary-score model."""
        return parse_binary_score_response(text, self.model_class)


def get_document_grader(llm: BaseChatModel):
    """Create a document relevance grader chain."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a grader assessing document relevance. "
            'Return only JSON: {{"binary_score": "yes"}} or '
            '{{"binary_score": "no"}}.',
        ),
        ("human", DOCUMENT_GRADER_PROMPT),
    ])
    return prompt | llm | BinaryScoreOutputParser(
        model_class=GradeDocumentRelevance,
    )


def get_hallucination_grader(llm: BaseChatModel):
    """Create a hallucination grader chain."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a grader assessing factual grounding. "
            'Return only JSON: {{"binary_score": "yes"}} or '
            '{{"binary_score": "no"}}.',
        ),
        ("human", HALLUCINATION_GRADER_PROMPT),
    ])
    return prompt | llm | BinaryScoreOutputParser(
        model_class=GradeHallucination,
    )


def get_answer_grader(llm: BaseChatModel):
    """Create an answer quality grader chain."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a grader assessing answer quality. "
            'Return only JSON: {{"binary_score": "yes"}} or '
            '{{"binary_score": "no"}}.',
        ),
        ("human", ANSWER_GRADER_PROMPT),
    ])
    return prompt | llm | BinaryScoreOutputParser(model_class=GradeAnswer)
