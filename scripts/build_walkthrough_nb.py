"""Generate a step-by-step Jupyter notebook walking through the wiki RAG pipeline.

Run with plain python (stdlib only):  python scripts/build_walkthrough_nb.py
Writes docs/writing/wiki_pipeline_walkthrough.ipynb. The notebook itself must be
executed where langconnect is importable and the DB/embeddings/LLM are reachable
(i.e. inside the langconnect-api container, or a host env with the same access).
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("docs/writing/wiki_pipeline_walkthrough.ipynb")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    md(
        "# LLM Wiki RAG — step-by-step walkthrough (real data)\n\n"
        "Traces how the two wiki levers work on one real MDA-QA question:\n"
        "- **PROMOTION** — adds wiki-pointed chunks to retrieval\n"
        "- **CONCEPT / INJECTION** — adds wiki summaries to the generation prompt\n\n"
        "**How to run:** execute inside the `langconnect-api` container (langconnect importable, "
        "DB + PubMedBERT embeddings + agent LLM reachable). For example, copy this file in and "
        "launch Jupyter there, or run the equivalent cells via `docker exec`.\n\n"
        "Example question (`id=0`) is one where summary **injection flipped the answer from wrong "
        "to right** under the controlled test."
    ),
    code(
        "import os, asyncio\n"
        "os.environ['PYTHONPATH'] = '/app'\n"
        "COLLECTION_ID = '28d0e6e0-99f1-4b03-b7ed-ebfdcd7371f1'  # mdaqa-pilot (281 papers)\n"
        "QUESTION_ID = '0'\n"
        "WIKI_DIR = '/app/llm_wiki/collections'\n\n"
        "from scripts.mdaqa_h1_ab import load_cases, judge_correct\n"
        "from langconnect.agent import run_agentic_search, _resolve_wiki_promotion\n"
        "from langconnect.agent.config import get_agent_llm\n"
        "from langconnect.agent.prompts import ANSWER_GENERATOR_PROMPT\n"
        "from langconnect.agent.wiki_context import resolve_wiki_context, _select_pages, _validate_pages\n"
        "from langconnect.database.collections import Collection\n"
        "from langchain_core.prompts import ChatPromptTemplate\n"
        "import json as _json\n"
    ),
    md(
        "## Step 0 — the question, gold answer, and gold evidence\n"
        "MDA-QA questions are multi-document: `support` lists the arXiv ids whose papers contain the answer."
    ),
    code(
        "cases = {c['id']: c for c in load_cases(50)}\n"
        "case = cases[QUESTION_ID]\n"
        "print('Q   :', case['question'])\n"
        "print('GOLD:', case['answer'][:300])\n"
        "print('SUPPORT (gold arxiv ids):', case['support'])\n"
        "print('-> needs', len(case['support']), 'documents (multi-hop)')\n"
    ),
    md(
        "## Step 1 — wiki page selection (the shared top-3)\n"
        "`resolve_wiki_context` loads the runtime pack and `_select_pages` keeps the top-3 pages most "
        "related to the question. **Both** levers use these same pages. Note which are `source` vs `concept`."
    ),
    code(
        "wiki = resolve_wiki_context(COLLECTION_ID, case['question'], wiki_dir=WIKI_DIR)\n"
        "print('status:', wiki.status, '| selected', len(wiki.selected_pages), 'pages\\n')\n"
        "for p in wiki.selected_pages:\n"
        "    kind = 'CONCEPT' if p['id'].startswith('concept-') else 'source'\n"
        "    print(f\"[{kind}] score={p.get('score')}  {p['title'][:70]}\")\n"
        "    print('   summary:', p['summary'][:160])\n"
        "    print('   source_refs (file_id:chunk_id):', [f\"{r['file_id'][:8]}:{r['chunk_id'][:8]}\" for r in p['source_refs'][:4]])\n"
    ),
    md(
        "## Step 1b — is top-3 enough? (coverage vs k)\n"
        "Vary the selection limit and count how many **distinct documents** the selected pages point to. "
        "Compare against the number of gold support docs. If the gold docs need more breadth than top-3 "
        "covers, raising `MAX_SELECTED_PAGES` (or giving concepts dedicated slots) is a real knob."
    ),
    code(
        "pack = _json.load(open(f'{WIKI_DIR}/{COLLECTION_ID}.json'))\n"
        "pages = _validate_pages(pack, COLLECTION_ID)\n"
        "for k in (3, 5, 10):\n"
        "    sel = _select_pages(pages, case['question'], limit=k)\n"
        "    files = {r['file_id'] for p in sel for r in p['source_refs']}\n"
        "    nconcept = sum(1 for p in sel if p['id'].startswith('concept-'))\n"
        "    print(f'top-{k:>2}: {len(sel)} pages ({nconcept} concept) -> {len(files)} distinct docs reachable')\n"
        "print('\\n(gold needs', len(case['support']), 'specific docs; more pages = more coverage but more noise)')\n"
    ),
    md(
        "## Step 2 — baseline retrieval (wiki OFF)\n"
        "Plain hybrid search, the chunks the agent would see with no wiki."
    ),
    code(
        "coll = Collection(collection_id=COLLECTION_ID)\n"
        "base_docs = await coll.search(case['question'], limit=5, search_type='hybrid')\n"
        "print(f'baseline retrieved {len(base_docs)} chunks:')\n"
        "for d in base_docs:\n"
        "    m = d.get('metadata', {})\n"
        "    print(f\"  score={d.get('score'):.3f} file={str(m.get('file_id'))[:8]} :: {(d.get('page_content') or '')[:80]}\")\n"
    ),
    md(
        "## Step 3 — PROMOTION (lever 1): add wiki-pointed chunks\n"
        "`_resolve_wiki_promotion` turns the top-3 pages into extra chunks (default: the frozen "
        "`source_refs`; with `WIKI_DOC_ROUTING=true` it re-searches inside the routed documents). "
        "These are **appended** to the baseline (dedup by id), then graded like any chunk."
    ),
    code(
        "src_refs, promoted, status, _ = await _resolve_wiki_promotion(COLLECTION_ID, wiki, case['question'])\n"
        "print('promotion status:', status, '| promoted', len(promoted), 'chunks')\n"
        "base_ids = {d.get('id') for d in base_docs}\n"
        "new = [d for d in promoted if d.get('id') not in base_ids]\n"
        "print(f'of those, {len(new)} are NEW (not already in baseline) -> the recall boost\\n')\n"
        "for d in promoted:\n"
        "    m = d.get('metadata', {})\n"
        "    tag = 'NEW' if d.get('id') not in base_ids else 'dup'\n"
        "    print(f\"  [{tag}] file={str(m.get('file_id'))[:8]} :: {(d.get('page_content') or '')[:80]}\")\n"
    ),
    md(
        "## Step 4 — INJECTION (lever 2): the summary block added to the prompt\n"
        "When `WIKI_CONTEXT_INJECT=true`, the generate step prepends this non-authoritative block "
        "(the top-3 summaries) before the retrieved context. This is the exact text injected."
    ),
    code(
        "print(wiki.context)\n"
    ),
    md(
        "## Step 5 — generation: OFF vs ON, on the SAME fixed context\n"
        "To isolate injection from pipeline noise, we fix the graded context (one agentic run) and "
        "generate twice: without the summary block (A) and with it (B). For `id=0`, B is correct and A is not."
    ),
    code(
        "res = await run_agentic_search(question=case['question'], collection_id=COLLECTION_ID,\n"
        "                               use_wiki_context=True, max_rewrites=1)\n"
        "relevant = res.get('relevant_documents', [])\n"
        "context = '\\n\\n---\\n\\n'.join(d.get('page_content','') for d in relevant)\n"
        "INJECT = ('Background orientation from a non-authoritative LLM Wiki '\n"
        "          '(navigation memory only -- do NOT cite it as a source; ground your '\n"
        "          'answer in the retrieved context below):\\n' + wiki.context +\n"
        "          '\\n\\n---\\n\\nRetrieved context:\\n' + context)\n\n"
        "llm = get_agent_llm()\n"
        "async def gen(ctx):\n"
        "    chain = ChatPromptTemplate.from_messages([('human', ANSWER_GENERATOR_PROMPT)]) | llm\n"
        "    return (await chain.ainvoke({'question': case['question'], 'context': ctx})).content\n\n"
        "ans_off = await gen(context)\n"
        "ans_on  = await gen(INJECT)\n"
        "print('A  (injection OFF):', ans_off[:300], '\\n')\n"
        "print('B  (injection ON ):', ans_on[:300], '\\n')\n"
        "print('GOLD:', case['answer'][:300])\n"
    ),
    code(
        "off_ok = await judge_correct(llm, case['question'], case['answer'], ans_off)\n"
        "on_ok  = await judge_correct(llm, case['question'], case['answer'], ans_on)\n"
        "print('judge — injection OFF correct:', off_ok)\n"
        "print('judge — injection ON  correct:', on_ok)\n"
    ),
    md(
        "## Summary — what each lever did\n\n"
        "| Lever | Stage | What it changed here |\n"
        "|---|---|---|\n"
        "| **Selection** | pre | top-3 pages chosen for the question (source + concept) |\n"
        "| **Promotion** | retrieval | appended NEW wiki-pointed chunks the base search missed |\n"
        "| **Injection** | generation | added the top-3 summaries as background -> flipped the answer |\n\n"
        "Toggles: `WIKI_SEMANTIC_SELECT` (selection), `WIKI_DOC_ROUTING` (promotion mode), "
        "`WIKI_CONTEXT_INJECT` (injection). Try other `QUESTION_ID`s (e.g. `'7'`, `'26'` show promotion "
        "recovering context) and flip the toggles to see each step move."
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {OUT} ({len(cells)} cells)")
