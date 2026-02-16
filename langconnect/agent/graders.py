"""Agentic RAG grader models.

Pydantic models used with LLM structured output to get binary yes/no
decisions for document relevance, hallucination detection, and answer quality.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langconnect.agent.prompts import (
    ANSWER_GRADER_PROMPT,
    DOCUMENT_GRADER_PROMPT,
    HALLUCINATION_GRADER_PROMPT,
)


class GradeDocumentRelevance(BaseModel):
    """Binary score for document relevance to a question."""

    binary_score: str = Field(
        description="Document relevance: 'yes' or 'no'"
    )


class GradeHallucination(BaseModel):
    """Binary score for whether generation is grounded in facts."""

    binary_score: str = Field(
        description="Answer grounded in facts: 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary score for whether answer addresses the question."""

    binary_score: str = Field(
        description="Answer addresses question: 'yes' or 'no'"
    )


def get_document_grader(llm: BaseChatModel):
    """Create a document relevance grader chain."""
    structured_llm = llm.with_structured_output(GradeDocumentRelevance)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing document relevance. Respond with structured output."),
        ("human", DOCUMENT_GRADER_PROMPT),
    ])
    return prompt | structured_llm


def get_hallucination_grader(llm: BaseChatModel):
    """Create a hallucination grader chain."""
    structured_llm = llm.with_structured_output(GradeHallucination)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing factual grounding. Respond with structured output."),
        ("human", HALLUCINATION_GRADER_PROMPT),
    ])
    return prompt | structured_llm


def get_answer_grader(llm: BaseChatModel):
    """Create an answer quality grader chain."""
    structured_llm = llm.with_structured_output(GradeAnswer)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing answer quality. Respond with structured output."),
        ("human", ANSWER_GRADER_PROMPT),
    ])
    return prompt | structured_llm
