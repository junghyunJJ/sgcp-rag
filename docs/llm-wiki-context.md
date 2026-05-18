# LLM Wiki Context for Agentic RAG

LLM Wiki context is an optional generation-time navigation layer. It is not a source of truth, not citation evidence, and not part of retrieval or query rewriting.

## Contract

- Request flag: `use_wiki_context`, default `false`.
- Pack path: `llm_wiki/collections/{collection_id}.json`.
- Override directory: `LANGCONNECT_WIKI_CONTEXT_DIR`.
- Maximum selected pages: 3.
- Selection output is metadata only: page id, title, summary, score, and navigation `source_refs`.
- Grounding evidence remains `relevant_documents`; hallucination grading must not include wiki context.
- The runtime JSON pack is the public activation signal. Markdown pages, `manifest.json`, and `log.md` are generated inspection artifacts.

## Pack Schema

```json
{
  "collection_id": "collection-uuid",
  "pages": [
    {
      "id": "stagent",
      "title": "STAgent",
      "summary": "Interprets pancreatic beta cell maturation.",
      "keywords": ["single-cell", "biological interpretation"],
      "source_refs": [
        {"file_id": "paper-a", "chunk_id": "chunk-a"}
      ]
    }
  ]
}
```

All page fields above are required. `source_refs` are navigation breadcrumbs back to retrieved material; they do not make a wiki page authoritative.

## Generated Rebuild Outputs

Manual and upload-triggered rebuilds write generated files under `llm_wiki/` only:

- `llm_wiki/collections/{collection_id}/sources/*.md`
- `llm_wiki/collections/{collection_id}/concepts/*.md`
- `llm_wiki/collections/{collection_id}/SCHEMA.md`
- `llm_wiki/collections/{collection_id}/index.md`
- `llm_wiki/collections/{collection_id}/log.md`
- `llm_wiki/collections/{collection_id}/manifest.json`
- `llm_wiki/collections/{collection_id}.json`

Source and concept Markdown pages include typed YAML frontmatter with `title`, `type`, `summary`, `keywords`, `source_refs`, `generated_at`, `updated_at`, and `confidence`. These files are replaceable generated artifacts; manual edits in these generated paths are not preserved by the MVP full rebuild.

`index.md` is a content catalog grouped by sources and concepts. `log.md` is the latest successful rebuild report, not append-only history. `manifest.json` records source/concept page metadata and the runtime pack path.

## Rebuild Surfaces

Manual rebuild:

```http
POST /collections/{collection_id}/llm-wiki/rebuild
```

Optional body fields follow the agentic-search LLM override pattern:

```json
{
  "llm_provider": "ollama",
  "llm_model": "qwen",
  "llm_temperature": 0
}
```

MCP stdio and SSE expose `rebuild_llm_wiki(collection_id, llm_provider=None, llm_model=None, llm_temperature=None)` and delegate to the REST endpoint.

Document upload runs a full rebuild after non-empty vector upsert success. If upsert succeeds but rebuild fails, vectors remain committed and upload returns HTTP 500 with stable partial-success detail:

```json
{
  "success": false,
  "error": "documents_indexed_wiki_rebuild_failed",
  "message": "Documents were indexed, but LLM Wiki rebuild failed.",
  "documents_indexed": true,
  "added_chunk_ids": ["..."],
  "wiki_rebuild_error": "..."
}
```

Retry recovery is the manual rebuild endpoint or MCP tool.

## Status Values

- `disabled`: request did not opt in.
- `selected`: one or more pages were selected and rendered.
- `missing_pack`: no JSON pack exists for the collection.
- `no_match`: pack exists, but no page matched the question.
- `invalid_json`: pack exists, but is not valid JSON.
- `invalid_schema`: collection id/path or pack shape is invalid.

## Runtime Behavior

`run_agentic_search` resolves wiki context before starting the graph. If pages are selected, `generate` uses a prompt that explicitly labels the wiki text as non-authoritative navigation memory. Retrieval, query rewriting, document relevance grading, hallucination grading, and answer grading continue to operate from the normal raw retrieved documents.

Rebuild publishes transactionally: Markdown and manifest artifacts are staged first, the runtime pack is schema-validated, then `llm_wiki/collections/{collection_id}.json` is replaced last. If validation or pre-commit publish fails, the previous public wiki remains visible to `agentic_search`.
