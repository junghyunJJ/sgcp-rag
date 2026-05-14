"""Agentic RAG graph node functions.

Five nodes that form the processing pipeline:
- retrieve: Fetches documents using existing Collection.search()
- grade_documents: Filters retrieved docs by LLM relevance scoring
- generate: Produces an answer from relevant documents
- rewrite_query: Rewrites the question for better retrieval
- grade_generation: Validates answer (hallucination + quality checks)
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from langconnect.agent.graders import (
    get_answer_grader,
    get_document_grader,
    get_hallucination_grader,
)
from langconnect.agent.prompts import ANSWER_GENERATOR_PROMPT, QUERY_REWRITER_PROMPT
from langconnect.agent.state import AgentState
from langconnect.database.collections import Collection

logger = logging.getLogger(__name__)


async def retrieve(state: AgentState) -> dict[str, Any]:
    """Retrieve documents using existing Collection.search().

    Calls the same search path as the regular /documents/search endpoint,
    so no search logic is duplicated.
    """
    logger.info("--- RETRIEVE ---")
    question = state["question"]
    collection_id = state["collection_id"]

    collection = Collection(collection_id=collection_id)
    documents = await collection.search(
        question,
        limit=state.get("search_limit", 5),
        search_type=state.get("search_type", "hybrid"),
        filter=state.get("search_filter"),
        min_score=state.get("min_score"),
    )

    return {
        "documents": documents,
        "steps": [f"retrieve: found {len(documents)} documents"],
    }


async def grade_documents(
    state: AgentState, llm: BaseChatModel,
) -> dict[str, Any]:
    """Grade each retrieved document for relevance to the question."""
    logger.info("--- GRADE DOCUMENTS ---")
    question = state["question"]
    documents = state.get("documents", [])
    grader = get_document_grader(llm)
    relevant_docs = []

    for doc in documents:
        content = doc.get("page_content", "")
        result = await grader.ainvoke(
            {"document": content, "question": question}
        )
        if result.binary_score.lower() == "yes":
            relevant_docs.append(doc)

    return {
        "relevant_documents": relevant_docs,
        "steps": [f"grade_documents: {len(relevant_docs)}/{len(documents)} relevant"],
    }


async def generate(
    state: AgentState, llm: BaseChatModel,
) -> dict[str, Any]:
    """Generate an answer from relevant documents."""
    logger.info("--- GENERATE ---")
    question = state["question"]
    relevant_docs = state.get("relevant_documents", [])
    context = "\n\n---\n\n".join(
        doc.get("page_content", "") for doc in relevant_docs
    )

    prompt = ChatPromptTemplate.from_messages([
        ("human", ANSWER_GENERATOR_PROMPT),
    ])
    chain = prompt | llm

    result = await chain.ainvoke({"question": question, "context": context})
    generation = result.content

    return {"generation": generation, "steps": ["generate: answer produced"]}


async def no_context(state: AgentState) -> dict[str, Any]:
    """Terminate the graph when retrieval cannot find relevant context."""
    logger.info("--- NO CONTEXT ---")
    return {
        "error": "no_relevant_context",
        "generation": "",
        "relevant_documents": [],
        "no_context_found": True,
        "steps": ["no_context: no relevant documents found"],
    }


async def rewrite_query(
    state: AgentState, llm: BaseChatModel,
) -> dict[str, Any]:
    """Rewrite the question for better vector search retrieval."""
    logger.info("--- REWRITE QUERY ---")
    question = state["question"]
    rewrite_count = state.get("rewrite_count", 0)

    prompt = ChatPromptTemplate.from_messages([
        ("human", QUERY_REWRITER_PROMPT),
    ])
    chain = prompt | llm

    result = await chain.ainvoke({"question": question})
    new_question = result.content

    rewrite_count += 1

    return {
        "question": new_question,
        "query_rewrites": [new_question],
        "rewrite_count": rewrite_count,
        "steps": [f"rewrite_query: '{question}' -> '{new_question}'"],
    }


async def grade_generation(
    state: AgentState, llm: BaseChatModel,
) -> dict[str, Any]:
    """Two-stage verification: hallucination check + answer quality check.

    Returns a steps update indicating pass/fail for each stage.
    The routing logic in graph.py uses the steps to decide next action.
    """
    logger.info("--- GRADE GENERATION ---")
    generation = state.get("generation", "")
    relevant_docs = state.get("relevant_documents", [])
    question = state["question"]

    # Stage 1: Hallucination check
    documents_text = "\n\n".join(
        doc.get("page_content", "") for doc in relevant_docs
    )
    hallucination_grader = get_hallucination_grader(llm)
    hallucination_result = await hallucination_grader.ainvoke(
        {"documents": documents_text, "generation": generation}
    )

    if hallucination_result.binary_score.lower() != "yes":
        return {"steps": ["grade_generation: FAILED hallucination check"]}

    # Stage 2: Answer quality check
    answer_grader = get_answer_grader(llm)
    answer_result = await answer_grader.ainvoke(
        {"question": question, "generation": generation}
    )

    if answer_result.binary_score.lower() != "yes":
        return {"steps": ["grade_generation: FAILED answer quality check"]}

    return {"steps": ["grade_generation: PASSED both checks"]}
