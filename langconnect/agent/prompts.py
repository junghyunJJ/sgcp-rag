"""Agentic RAG prompt templates.

Five prompts that drive the agent's decision-making loop:
1. Document grading — is a retrieved document relevant to the question?
2. Query rewriting — improve the query for better vector search results
3. Answer generation — synthesize an answer from relevant documents
4. Hallucination grading — is the answer grounded in the documents?
5. Answer grading — does the answer actually address the question?
"""

DOCUMENT_GRADER_PROMPT = """\
You are a grader assessing the relevance of a retrieved document to a user question.

Retrieved document:
{document}

User question:
{question}

If the document contains keyword(s) or semantic meaning related to the question, \
grade it as relevant. Give a binary score: 'yes' or 'no' to indicate whether the \
document is relevant to the question."""

QUERY_REWRITER_PROMPT = """\
You are a question re-writer that converts an input question to a better version \
optimized for vector store retrieval. Look at the input and try to reason about \
the underlying semantic intent.

Here is the initial question:
{question}

Formulate an improved question."""

ANSWER_GENERATOR_PROMPT = """\
You are an assistant for question-answering tasks. Use the following pieces of \
retrieved context to answer the question. If you don't know the answer, just say \
that you don't know. Keep the answer concise (3-5 sentences max).

Question: {question}

Context:
{context}

Answer:"""

ANSWER_GENERATOR_WITH_WIKI_PROMPT = """\
You are an assistant for question-answering tasks. Use the retrieved context to \
answer the question. The LLM Wiki context is non-authoritative navigation memory: \
it may help orient interpretation, but it is not evidence and must not be cited or \
treated as support. If the retrieved context does not support the answer, just say \
that you don't know. Keep the answer concise (3-5 sentences max).

Question: {question}

LLM Wiki context:
{wiki_context}

Retrieved context:
{context}

Answer:"""

HALLUCINATION_GRADER_PROMPT = """\
You are a grader assessing whether an LLM generation is grounded in / supported by \
a set of retrieved facts. Give a binary score: 'yes' or 'no'. 'Yes' means that the \
answer is grounded in the set of facts.

Set of facts:
{documents}

LLM generation:
{generation}"""

ANSWER_GRADER_PROMPT = """\
You are a grader assessing whether an answer addresses / resolves a question. \
Give a binary score: 'yes' or 'no'. 'Yes' means that the answer resolves the question.

User question:
{question}

LLM generation:
{generation}"""
