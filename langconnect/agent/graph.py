"""Agentic RAG StateGraph construction.

Builds the LangGraph graph with conditional edges:

  START → retrieve → grade_documents → [relevant?]
                                          ├─ YES → generate → grade_generation → [passed?]
                                          │                                        ├─ YES → END
                                          │                                        └─ NO → rewrite_query
                                          └─ NO → rewrite_query
                                                        └──→ retrieve (loop, max N)
"""

import functools
import logging

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

from langconnect.agent.nodes import (
    generate,
    grade_documents,
    grade_generation,
    no_context,
    retrieve,
    rewrite_query,
)
from langconnect.agent.state import AgentState

logger = logging.getLogger(__name__)


def _route_after_grading(state: AgentState) -> str:
    """Decide next step after document grading."""
    relevant_docs = state.get("relevant_documents", [])
    if relevant_docs:
        return "generate"
    # No relevant docs — rewrite if under the limit
    rewrite_count = state.get("rewrite_count", 0)
    max_rewrites = state.get("max_rewrites", 3)
    if rewrite_count >= max_rewrites:
        logger.warning("Max rewrites reached with no relevant docs. Ending no-context.")
        return "no_context"
    return "rewrite_query"


def _route_after_generation_check(state: AgentState) -> str:
    """Decide next step after generation quality checks."""
    steps = state.get("steps", [])
    last_step = steps[-1] if steps else ""

    if "PASSED" in last_step:
        return END

    # Failed a check — rewrite if under the limit
    rewrite_count = state.get("rewrite_count", 0)
    max_rewrites = state.get("max_rewrites", 3)
    if rewrite_count >= max_rewrites:
        logger.warning("Max rewrites reached. Returning current generation.")
        return END
    return "rewrite_query"


def build_agentic_rag_graph(llm: BaseChatModel) -> StateGraph:
    """Build and compile the Agentic RAG graph.

    Args:
        llm: The language model to use for grading and generation.

    Returns:
        A compiled LangGraph StateGraph.
    """
    graph = StateGraph(AgentState)

    # Bind LLM to node functions that need it (retrieve doesn't need LLM)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", functools.partial(grade_documents, llm=llm))
    graph.add_node("generate", functools.partial(generate, llm=llm))
    graph.add_node("no_context", no_context)
    graph.add_node("rewrite_query", functools.partial(rewrite_query, llm=llm))
    graph.add_node("grade_generation", functools.partial(grade_generation, llm=llm))

    # Edges
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        _route_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "no_context": "no_context",
        },
    )

    graph.add_edge("generate", "grade_generation")
    graph.add_edge("no_context", END)

    graph.add_conditional_edges(
        "grade_generation",
        _route_after_generation_check,
        {END: END, "rewrite_query": "rewrite_query"},
    )

    graph.add_edge("rewrite_query", "retrieve")

    return graph.compile()
