"""Tests for Agentic RAG binary grader parsing."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langconnect.agent.graders import (
    GradeAnswer,
    GradeDocumentRelevance,
    get_answer_grader,
    get_document_grader,
    get_hallucination_grader,
    parse_binary_score_response,
)


def test_parse_binary_score_response_accepts_json_object():
    """Binary graders should accept schema-shaped JSON output."""
    result = parse_binary_score_response(
        '{"binary_score": "yes"}',
        GradeDocumentRelevance,
    )

    assert result.binary_score == "yes"


def test_parse_binary_score_response_accepts_bare_yes_no():
    """Qwen/Ollama may return a bare yes/no instead of structured JSON."""
    yes_result = parse_binary_score_response("yes", GradeDocumentRelevance)
    no_result = parse_binary_score_response("No.", GradeAnswer)

    assert yes_result.binary_score == "yes"
    assert no_result.binary_score == "no"


def test_parse_binary_score_response_rejects_ambiguous_text():
    """Ambiguous grader output should fail instead of guessing."""
    with pytest.raises(ValueError, match="binary_score"):
        parse_binary_score_response("yes, but maybe no", GradeDocumentRelevance)


@pytest.mark.parametrize(
    ("factory", "payload"),
    [
        (
            get_document_grader,
            {"document": "CellAgent automates scRNA-seq analysis.", "question": "What is CellAgent?"},
        ),
        (
            get_hallucination_grader,
            {"documents": "CellAgent automates scRNA-seq analysis.", "generation": "CellAgent automates scRNA-seq analysis."},
        ),
        (
            get_answer_grader,
            {"question": "What is CellAgent?", "generation": "CellAgent automates scRNA-seq analysis."},
        ),
    ],
)
def test_binary_graders_escape_json_examples(factory, payload):
    """JSON examples in grader prompts must not be interpreted as template vars."""
    llm = FakeListChatModel(responses=['{"binary_score": "yes"}'])
    result = factory(llm).invoke(payload)

    assert result.binary_score == "yes"
