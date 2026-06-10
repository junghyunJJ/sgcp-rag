# SGCP-RAG

> **An Agentic RAG system with a self-rebuilding LLM Wiki navigation layer, MCP integration, and a Next.js GUI.**

`SGCP-RAG` is a full-stack RAG (Retrieval-Augmented Generation) system built on PostgreSQL + `pgvector`. You upload documents into collections; they are automatically parsed, chunked, embedded, and searchable. On top of that, a **self-rebuilding LLM Wiki** maintains a markdown knowledge base that acts as a navigation layer for an **Agentic / Adaptive RAG** search graph. Everything is exposed simultaneously via a Next.js UI, a REST API, and an MCP (Model Context Protocol) server.

This project is a fork of [`teddynote-lab/langconnect-client`](https://github.com/teddynote-lab/langconnect-client) (itself based on LangChain AI's [`langconnect`](https://github.com/langchain-ai/langconnect)) extended with:

- a LangGraph-based **Agentic RAG** self-correcting search graph,
- a **per-collection LLM Wiki** inspired by [Karpathy's gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f),
- **Ollama-first LLM routing** with graceful OpenAI/Google fallback,
- domain-specific **PubMedBERT** embeddings.

---

## Table of Contents

1. [Overview — What this project does](#1-overview--what-this-project-does)
2. [Agentic RAG and Adaptive RAG](#2-agentic-rag-and-adaptive-rag)
3. [LLM Wiki — Karpathy's idea, adapted for RAG](#3-llm-wiki--karpathys-idea-adapted-for-rag)
4. [Architecture](#4-architecture)
5. [Quick Start](#5-quick-start)
6. [Usage in Detail](#6-usage-in-detail)
   - [6.1 Using the Web UI](#61-using-the-web-ui)
   - [6.2 Using the REST API](#62-using-the-rest-api)
   - [6.3 The Agentic Search API](#63-the-agentic-search-api)
   - [6.4 The LLM Wiki API](#64-the-llm-wiki-api)
   - [6.5 Using the MCP server](#65-using-the-mcp-server)
7. [Environment variables](#7-environment-variables)
8. [Testing](#8-testing)
9. [License](#9-license)

---

## 1. Overview — What this project does

`SGCP-RAG` bundles three responsibilities into one system:

| Capability | What it does |
|------------|--------------|
| **RAG infrastructure** | Upload PDF / DOCX / HTML / TXT / MD → parse with PyMuPDF4LLM → chunk → embed with PubMedBERT (768-dim) → store in PostgreSQL `pgvector` → search via semantic / keyword / hybrid. |
| **Agentic RAG search** | A LangGraph `StateGraph` runs a self-correcting loop: retrieve → grade documents → generate → grade for hallucination & answer quality → rewrite the query and retry on failure. |
| **LLM Wiki metadata layer** | An LLM automatically summarises and categorises every document in a collection into markdown pages (`llm_wiki/collections/{collection_id}/`) plus a runtime JSON pack. Rebuilds happen automatically on upload/delete. Agentic search optionally uses it as a navigation hint. |

Three independent entry points access the same backend:

- **Next.js Web UI** — `http://localhost:3005`
- **REST API** — `http://localhost:8888` (Swagger UI at `/docs`)
- **MCP server** — `mcpserver/mcp_server.py` (stdio) or `mcpserver/mcp_sse_server.py` (SSE on port 8765)

---

## 2. Agentic RAG and Adaptive RAG

### 2.1 The problem with naive RAG

A traditional RAG pipeline is unidirectional:

```
question → retrieve → generate → answer
```

This has two well-known weaknesses:

- Retrieved documents that are **irrelevant to the question** are still fed to the generator, degrading answer quality.
- If the LLM **hallucinates content not present in the documents**, there is no detection mechanism.

### 2.2 The Adaptive RAG idea

Adaptive RAG (the [LangChain `07-LangGraph-Adaptive-RAG`](https://github.com/teddynote-lab/langchain-kr) notebook family) addresses this by introducing **conditional routing and self-grading loops** into the graph:

```
START → route_question → retrieve → grade_documents
                                      ├─ relevant → generate → hallucination_check
                                      │                          ├─ pass → END
                                      │                          └─ fail → transform_query → retrieve
                                      └─ not relevant → transform_query → retrieve
```

Three core ideas:

1. **Document relevance grading** — each retrieved chunk is shown to an LLM with a binary `yes/no` prompt; only the passing chunks are handed to the generator.
2. **Hallucination grading** — after generation, the LLM judges whether the answer is *grounded in the retrieved documents*.
3. **Query rewriting** — when either check fails, the question is rewritten to be more retrieval-friendly and the loop restarts from `retrieve`.

### 2.3 The Agentic RAG in this project

This project adopts the Adaptive RAG self-correcting skeleton, **simplifies it for a single-vectorstore environment**, and **adds production safety guards**. The implementation lives in `langconnect/agent/`.

```
                    ┌──────────┐
                    │  START   │
                    └────┬─────┘
                         ▼
                  ┌─────────────┐
                  │  retrieve   │  ← Collection.search() (hybrid by default)
                  └──────┬──────┘   + (optional) LLM Wiki source_ref promotion
                         ▼
                  ┌─────────────────┐
                  │ grade_documents │  ← per-document LLM yes/no
                  └────────┬────────┘
                           │
                ┌──────────┴───────────┐
        relevant > 0              relevant == 0
                │                       │
                ▼                       ▼
         ┌──────────┐          ┌──────────────┐
         │ generate │          │rewrite_query │  rewrite_count++
         └────┬─────┘          └──────┬───────┘
              ▼                       │
       ┌──────────────────┐           │
       │ grade_generation │           │
       │  ① hallucination │           │
       │  ② answer quality│           │
       └─────┬────────┬───┘           │
             │        │               │
          PASSED   FAILED             │
             │        └───────────────┤
             ▼                        │
            END                       └───► retrieve (loop)

Loop guard: if rewrite_count >= max_rewrites, force-terminate.
```

Differences from the original Adaptive RAG notebook:

| Change | Why |
|--------|-----|
| Removed the `route_question` (web vs vectorstore) branch. | MCP clients are assumed to bring their own web search. The graph stays inside a single trust boundary: the vectorstore. |
| Added an explicit `rewrite_count` / `max_rewrites` counter. | Replaces the notebook's reliance on `recursion_limit` so we avoid `GraphRecursionError` and keep a hard token budget. |
| Separated `relevant_documents` from `documents` in state. | The notebook overwrites `documents` after filtering; we keep both so the trace can show what was filtered out. |
| Bound the LLM with `functools.partial(node, llm=llm)`. | Avoids the LangGraph state-serialisation issue you get from stuffing an LLM instance into state, and lets tests inject mocks trivially. |
| Reused the existing `Collection.search()` instead of a new retriever tool. | 100% reuse of the existing hybrid-search infrastructure — zero retrieval-code duplication. |

> For the deeper mapping see `docs/design-decisions-agentic-rag.md`; for the full graph see `docs/agenticRAG_architecture.md`.

---

## 3. LLM Wiki — Karpathy's idea, adapted for RAG

### 3.1 The original idea

In [his gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f), Andrej Karpathy proposes the following:

> *While an LLM works on a task, have it continuously write down what it just learned — recurring patterns, gotchas, useful shortcuts — into a markdown wiki. Inject that wiki back as context in future sessions and the assistant becomes incrementally smarter. In short, **a living human-readable knowledge base for an AI coding assistant**.*

The appeal is that (1) the artefact is plain markdown and stays human-inspectable, (2) humans can audit and edit it, and (3) it is a **concept-level navigation index** rather than another opaque embedding store.

### 3.2 How this project applies the idea

We bring Karpathy's wiki idea down to the **RAG collection** level — one wiki per collection.

```
llm_wiki/
└── collections/
    └── {collection_id}/
        ├── sources/           # one summary page per original document (.md + YAML frontmatter)
        ├── concepts/          # cross-cutting concepts extracted by the LLM (.md)
        ├── SCHEMA.md          # frontmatter schema documentation
        ├── index.md           # catalog grouped by sources + concepts
        ├── manifest.json      # page metadata + runtime pack path
        └── log.md             # report of the latest successful rebuild
    {collection_id}.json       # ← runtime JSON pack consumed by agentic_search
```

**Automatic rebuild triggers:**

- Right after a successful document upload (`POST /collections/{id}/documents`).
- Right after a document deletion (`DELETE /collections/{id}/documents/...`).
- On explicit demand (`POST /collections/{id}/llm-wiki/rebuild` or the MCP tool `rebuild_llm_wiki`).

### 3.3 The key design decision — wiki is *navigation*, not *evidence*

Karpathy's original wiki is injected directly into the prompt and the LLM treats it as ground truth. Doing that naively inside a RAG system would be a disaster: the wiki's *summaries* would silently become the evidence backing each answer, and your citation / hallucination grading would no longer reflect reality. We deliberately rule that out (`docs/llm-wiki-context.md`):

> *"LLM Wiki context is an optional navigation layer. It is not a source of truth, not citation evidence, and not a raw generation prompt channel."*

Instead, we only trust the wiki page's `source_refs` — coordinates back to real chunks.

```
By default, when `use_wiki_context` is not set or is `true`:

question ─┬─► wiki page selection  (up to 3 pages)
          │      └─► source_refs (file_id + chunk_id), capped at 8
          │              └─► re-fetch the real chunks from this collection (best-effort)
          │                      └─► append to retrieve() results
          │                          (dedup by chunk_id; normal retrieval wins collisions)
          ▼
   retrieve → grade_documents → generate → grade_generation → END

⚠️  generate() and the hallucination grader only see promoted *real* chunks.
    Wiki page titles / summaries / keywords are returned as response metadata only —
    they never enter the evidence set.
```

The net effect:

- The wiki acts as a **human-readable index** describing what the collection covers.
- It simultaneously serves as a **retrieval-boost signal**: even when a question uses different vocabulary than the source text, the concept page can pull in the right chunk.
- The hallucination grader always operates on a consistent set of raw chunks.

### 3.4 Page schema (summary)

Each wiki page (`sources/*.md`, `concepts/*.md`) carries a typed YAML frontmatter:

```markdown
---
title: STAgent
type: concept                    # source | concept
summary: Interprets pancreatic beta cell maturation across timepoints.
keywords: [single-cell, biological interpretation]
source_refs:
  - {file_id: paper-a, chunk_id: chunk-a-3}
  - {file_id: paper-b, chunk_id: chunk-b-7}
generated_at: 2026-05-19T12:34:00Z
updated_at: 2026-05-19T12:34:00Z
confidence: medium               # low | medium | high
---

(LLM-generated markdown body)
```

The runtime JSON pack (`llm_wiki/collections/{id}.json`) is a compressed index built from those frontmatters so `agentic_search` can do fast page selection without parsing markdown.

---

## 4. Architecture

```mermaid
graph TB
    subgraph "Client"
        UI["Next.js UI<br/>localhost:3005"]
        CD["Claude Desktop /<br/>MCP stdio client"]
        SSE["Web MCP / SSE client"]
    end

    subgraph "Application"
        API["FastAPI<br/>localhost:8888"]
        MCPS["mcp_server.py<br/>(stdio)"]
        MCPSSE["mcp_sse_server.py<br/>(SSE :8765)"]
        AGENT["Agentic RAG<br/>(langconnect/agent/)"]
        WIKI["LLM Wiki service<br/>(langconnect/services/llm_wiki.py)"]
    end

    subgraph "Data"
        PG["PostgreSQL 16<br/>+ pgvector"]
        FS["llm_wiki/ filesystem"]
    end

    subgraph "External"
        EMB["HuggingFace<br/>PubMedBERT (768d)"]
        OLLAMA["Ollama (local LLM)"]
        OAI["OpenAI / Google<br/>(fallback LLM)"]
    end

    UI --> API
    CD --> MCPS
    SSE --> MCPSSE
    MCPS --> API
    MCPSSE --> API
    API --> AGENT
    API --> WIKI
    AGENT --> PG
    AGENT --> OLLAMA
    AGENT --> OAI
    WIKI --> PG
    WIKI --> OLLAMA
    WIKI --> OAI
    WIKI --> FS
    API --> EMB
    API --> PG
```

| Component | Location |
|-----------|----------|
| FastAPI server | `langconnect/server.py` |
| Collections / Documents API | `langconnect/api/collections.py`, `langconnect/api/documents.py` |
| Agentic Search API | `langconnect/api/agentic.py` |
| LLM Wiki API | `langconnect/api/llm_wiki.py` |
| Agentic RAG graph | `langconnect/agent/{state,nodes,graders,prompts,graph,wiki_context}.py` |
| LLM Wiki rebuild service | `langconnect/services/llm_wiki.py` |
| Document parsing & chunking | `langconnect/services/document_processor.py`, `langconnect/parsers/` |
| MCP stdio server | `mcpserver/mcp_server.py` |
| MCP SSE server | `mcpserver/mcp_sse_server.py` |
| Next.js frontend | `next-connect-ui/` |

For full diagrams and sequence flows, see `docs/architecture-overview.md`, `docs/agenticRAG_architecture.md`, and `docs/llm-wiki-context.md`.

---

## 5. Quick Start

### 5.1 Prerequisites

- Docker & Docker Compose
- (optional) Node.js 20+ — for the MCP Inspector or local frontend dev
- (optional) Python 3.11+ with `uv` — for local backend dev or running the MCP stdio server outside Docker
- (optional) [Ollama](https://ollama.com/) — for local LLM inference (the agent automatically falls back to OpenAI/Google if Ollama is unreachable)

### 5.2 Configure `.env`

```bash
cp .env.example .env
```

At minimum, fill in:

```dotenv
# Embeddings (PubMedBERT runs locally; OPENAI_API_KEY is only needed for OpenAI fallback)
OPENAI_API_KEY=sk-...

# Ollama (preferred when available)
QUERY_EXPANSION_LLM_BASE_URL=http://localhost:11434
QUERY_EXPANSION_LLM_MODEL=qwen3.5:9b
AGENT_LLM_BASE_URL=http://localhost:11434
AGENT_LLM_PROVIDER=auto              # auto = try Ollama first, fall back to OpenAI
AGENT_LLM_MODEL=qwen3.5:122b
AGENT_LLM_OPENAI_MODEL=gpt-5.4

# PostgreSQL
POSTGRES_USER=llmwiki
POSTGRES_PASSWORD=llmwiki
POSTGRES_DB=llmwiki_rag_db

# Next.js
NEXTAUTH_SECRET=change-me
```

A complete variable reference is in [§7](#7-environment-variables).

### 5.3 Build and run

```bash
make build       # build the Next.js bundle + Docker images
make up          # start postgres + api + nextjs
make down        # stop everything
make restart     # bounce the stack
docker-compose logs -f   # tail logs
```

Once up:

| Service | URL |
|---|---|
| Next.js UI | http://localhost:3005 |
| API | http://localhost:8888 |
| API docs (Swagger) | http://localhost:8888/docs |
| Health check | http://localhost:8888/health |
| Postgres | localhost:5432 |

### 5.4 Generate the MCP config

```bash
make mcp
```

This writes `mcpserver/mcp_config.json`. Paste its contents into Claude Desktop / Cursor's MCP settings.

---

## 6. Usage in Detail

### 6.1 Using the Web UI

1. Open `http://localhost:3005`.
2. **Collections** page → create a new collection (name + optional metadata).
3. Open the collection → **Documents** tab → drag-and-drop PDF/MD/DOCX/HTML/TXT files.
   - The LLM Wiki rebuild starts automatically after the upload commits.
   - If embedding succeeds but wiki rebuild fails, the API returns HTTP 500 with `documents_indexed_wiki_rebuild_failed`. **The vectors stay committed** — you only need to retry the rebuild (see §6.4).
4. **Search** page → try `semantic` / `keyword` / `hybrid` search.
5. **LLM Wiki Viewer** → browse the auto-generated `sources/` and `concepts/` markdown pages.

### 6.2 Using the REST API

The full schema is interactive at `http://localhost:8888/docs`. The most useful calls:

```bash
# Create a collection
curl -X POST http://localhost:8888/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "papers", "metadata": {"topic": "spatial-transcriptomics"}}'

# Upload a document (multipart)
curl -X POST http://localhost:8888/collections/<COLLECTION_ID>/documents \
  -F "files=@./paper.pdf" \
  -F 'metadata={"source":"paper.pdf"}'

# Plain (non-agentic) search — hybrid
curl -X POST http://localhost:8888/collections/<COLLECTION_ID>/documents/search \
  -H "Content-Type: application/json" \
  -d '{"query":"how does beta cell maturation occur?",
       "limit":5,
       "search_type":"hybrid"}'
```

### 6.3 The Agentic Search API

`POST /collections/{collection_id}/agentic-search`

```bash
curl -X POST http://localhost:8888/collections/<COLLECTION_ID>/agentic-search \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main interaction pathways in pancreatic islet differentiation?",
    "search_type": "hybrid",
    "search_limit": 5,
    "max_rewrites": 3,
    "use_wiki_context": true,
    "llm_provider": "auto",
    "llm_model": "qwen3.5:122b",
    "llm_temperature": 0
  }'
```

Response:

```json
{
  "generation": "...LLM answer...",
  "relevant_documents": [
    {"id": "...", "page_content": "...", "metadata": {...}, "score": 0.82}
  ],
  "query_rewrites": ["v2 rewritten...", "v3 rewritten..."],
  "rewrite_count": 2,
  "steps": [
    "retrieve: found 5 documents",
    "grade_documents: 3/5 relevant",
    "generate: answer produced",
    "grade_generation: PASSED both checks"
  ],
  "wiki": {
    "status": "selected",
    "selected_pages": [
      {"id": "stagent", "title": "STAgent", "summary": "...", "score": 0.91}
    ],
    "promotion_status": "promoted",
    "promoted_chunk_count": 4
  }
}
```

Request parameters:

| Field | Default | Description |
|-------|---------|-------------|
| `question` | (required) | Natural-language question |
| `search_type` | `hybrid` | `semantic` / `keyword` / `hybrid` |
| `search_limit` | `5` | Number of chunks to retrieve per round |
| `search_filter` | `null` | Metadata filter (JSON object) |
| `max_rewrites` | `3` | Maximum number of query rewrites (loop guard) |
| `use_wiki_context` | `true` | Use existing LLM Wiki navigation context when available; set `false` to disable |
| `llm_provider` | env default | `auto` / `ollama` / `openai` / `google` |
| `llm_model` | env default | Model name override |
| `llm_temperature` | env default | Temperature override |

### 6.4 The LLM Wiki API

```bash
# Index (sources + concepts list)
curl http://localhost:8888/collections/<COLLECTION_ID>/llm-wiki

# Render a single markdown page
curl http://localhost:8888/collections/<COLLECTION_ID>/llm-wiki/sources/<page-slug>
curl http://localhost:8888/collections/<COLLECTION_ID>/llm-wiki/concepts/<page-slug>

# Force a rebuild (e.g. to recover from a failed auto-rebuild)
curl -X POST http://localhost:8888/collections/<COLLECTION_ID>/llm-wiki/rebuild \
  -H "Content-Type: application/json" \
  -d '{"llm_provider":"ollama","llm_model":"qwen3.5:122b","llm_temperature":0}'
```

Rebuilds are transactional: markdown + manifest artefacts are staged first → the runtime pack is schema-validated → finally `llm_wiki/collections/{collection_id}.json` is swapped in. If any step fails, **the previous wiki remains visible to `agentic_search`.**

### 6.5 Using the MCP server

#### 6.5.1 stdio server (Claude Desktop / Cursor)

```bash
make mcp                                # print mcp_config.json
uv run python mcpserver/mcp_server.py   # run manually (the MCP client usually auto-spawns this)
```

Example `mcp_config.json` to paste into Claude Desktop's MCP settings:

```json
{
  "mcpServers": {
    "sgcp-rag": {
      "command": "/path/to/uv",
      "args": ["run", "python", "/abs/path/to/mcpserver/mcp_server.py"],
      "env": {
        "API_BASE_URL": "http://localhost:8888",
        "OLLAMA_BASE_URL": "http://localhost:11434"
      }
    }
  }
}
```

#### 6.5.2 SSE server (web-based MCP clients)

```bash
./run_mcp_sse.sh                          # checks env, auto-authenticates, starts
# or
uv run python mcpserver/mcp_sse_server.py
```

- On startup it validates the existing token, prompting for email/password (and updating `.env`) if it has expired.
- Default port `8765` — override with `SSE_PORT`.
- Debug with the MCP Inspector:

```bash
npx @modelcontextprotocol/inspector
# → Transport: SSE
# → URL: http://localhost:8765/sse
```

#### 6.5.3 Available MCP tools

| Tool | Description |
|------|-------------|
| `list_collections` | List every collection |
| `get_collection` | Inspect one collection |
| `create_collection` | Create a new collection |
| `delete_collection` | Delete a collection |
| `list_documents` | List documents in a collection |
| `add_documents` | Add text documents |
| `add_documents_from_files` | Upload files from the filesystem (stdio server only) |
| `delete_document` | Delete a document |
| `search_documents` | Plain search (semantic / keyword / hybrid) |
| `multi_query` | Expand one question into multiple sub-queries via an LLM |
| `agentic_search` | The LangGraph self-correcting RAG search |
| `rebuild_llm_wiki` | Manually rebuild a collection's wiki |
| `get_health_status` | API health check |

#### 6.5.4 Suggested RAG prompt (Claude Desktop)

```markdown
You are a question-answer assistant grounded in the user's RAG collection.

#Steps:
1. Use `list_collections` to identify the right collection.
2. For focused questions, use `agentic_search` (it self-corrects in one call).
3. For exploratory questions, use `multi_query` to generate sub-questions
   and run each through `search_documents` (hybrid).
4. Always cite sources (file_id, page numbers) at the end.

#Format:
(answer)

**Sources**
- [1] file_id, page
- [2] ...
```

---

## 7. Environment variables

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `OPENAI_API_KEY` | — | △ | Needed for OpenAI embeddings or as the LLM fallback |
| `POSTGRES_HOST` | `postgres` | ✗ | Inside Docker this is the service name |
| `POSTGRES_PORT` | `5432` | ✗ |  |
| `POSTGRES_USER` | `llmwiki` | ✗ |  |
| `POSTGRES_PASSWORD` | `llmwiki` | ✓ | Change in production |
| `POSTGRES_DB` | `llmwiki_rag_db` | ✗ |  |
| `API_BASE_URL` | `http://localhost:8888` | ✗ | Used by the MCP server to reach the API |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8888` | ✓ | Used by the browser to reach the API |
| `NEXTAUTH_SECRET` | — | ✓ | NextAuth JWT signing key |
| `NEXTAUTH_URL` | `http://localhost:3005` | ✓ |  |
| `ALLOW_ORIGINS` | `["*"]` | ✗ | CORS allow-list (JSON array) |
| `IS_TESTING` | `false` | ✗ | Bypass-auth mode |
| `SSE_PORT` | `8765` | ✗ | MCP SSE server port |
| `OLLAMA_BASE_URL` | `http://localhost:5000` | ✗ | Shared Ollama fallback endpoint |
| `AGENT_LLM_BASE_URL` | (= `OLLAMA_BASE_URL`) | ✗ | Dedicated Ollama endpoint for Agentic RAG |
| `QUERY_EXPANSION_LLM_BASE_URL` | (= `OLLAMA_BASE_URL`) | ✗ | Dedicated Ollama endpoint for query expansion |
| `AGENT_LLM_PROVIDER` | `auto` | ✗ | `auto` / `ollama` / `openai` / `google` |
| `AGENT_LLM_MODEL` | `qwen3.5:122b` | ✗ | Ollama model name |
| `AGENT_LLM_OPENAI_MODEL` | `gpt-5.4` | ✗ | OpenAI fallback model |
| `AGENT_LLM_TEMPERATURE` | `0` | ✗ |  |
| `AGENT_MAX_REWRITES` | `3` | ✗ | Agentic loop guard |
| `QUERY_EXPANSION_LLM_PROVIDER` | `auto` | ✗ |  |
| `QUERY_EXPANSION_LLM_MODEL` | `qwen3.5:35b` | ✗ |  |
| `QUERY_EXPANSION_OPENAI_MODEL` | `gpt-5.4` | ✗ |  |
| `LANGCONNECT_WIKI_CONTEXT_DIR` | `llm_wiki/collections` | ✗ | Override directory for the runtime wiki pack |

> When `AGENT_LLM_PROVIDER=auto`, the agent first tries Ollama at `AGENT_LLM_BASE_URL` (or `OLLAMA_BASE_URL`), and on failure does **one** fallback to `AGENT_LLM_OPENAI_MODEL`. If you explicitly set `AGENT_LLM_PROVIDER=ollama`, no fallback happens — failures are propagated as-is.

---

## 8. Testing

```bash
# Everything
make test

# A single file
make test TEST_FILE=tests/unit_tests/test_documents_api.py

# pytest directly
uv run pytest tests/unit_tests -v

# Agentic RAG only
uv run pytest --confcutdir=tests/unit_tests \
  tests/unit_tests/test_agent_config.py \
  tests/unit_tests/test_agentic_search.py -v
```

Frontend:

```bash
cd next-connect-ui
npm test                # Jest
npm run test:watch
```

---

## 9. License

MIT — see [LICENSE](./LICENSE).

This project is a fork of [`teddynote-lab/langconnect-client`](https://github.com/teddynote-lab/langconnect-client) (in turn based on LangChain AI's [`langconnect`](https://github.com/langchain-ai/langconnect)). The Agentic RAG graph, the LLM Wiki layer, the Ollama-first LLM routing, and the PubMedBERT embedding integration are additions in this fork.

### References

- Karpathy, A. ["LLM Wiki" gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).
- LangGraph Adaptive / Agentic RAG notebooks — see the reference table in `docs/design-decisions-agentic-rag.md` §7.
- PubMedBERT: Gu Y. et al. *Domain-specific language model pretraining for biomedical NLP* (2021).
- PyMuPDF4LLM: https://github.com/pymupdf/PyMuPDF4llm
- Internal design docs:
  - `docs/architecture-overview.md`
  - `docs/agenticRAG_architecture.md`
  - `docs/llm-wiki-context.md`
  - `docs/design-decisions-agentic-rag.md`
