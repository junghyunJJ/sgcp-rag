# LLM Wiki Context for Agentic RAG

LLM Wiki context is an optional generation-time navigation layer. It is not a source of truth, not citation evidence, and not part of retrieval or query rewriting.

## Contract

- Request flag: `use_wiki_context`, default `false`.
- Pack path: `omx_wiki/collections/{collection_id}.json`.
- Override directory: `LANGCONNECT_WIKI_CONTEXT_DIR`.
- Maximum selected pages: 3.
- Selection output is metadata only: page id, title, summary, score, and navigation `source_refs`.
- Grounding evidence remains `relevant_documents`; hallucination grading must not include wiki context.

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

## Status Values

- `disabled`: request did not opt in.
- `selected`: one or more pages were selected and rendered.
- `missing_pack`: no JSON pack exists for the collection.
- `no_match`: pack exists, but no page matched the question.
- `invalid_json`: pack exists, but is not valid JSON.
- `invalid_schema`: collection id/path or pack shape is invalid.

## Runtime Behavior

`run_agentic_search` resolves wiki context before starting the graph. If pages are selected, `generate` uses a prompt that explicitly labels the wiki text as non-authoritative navigation memory. Retrieval, query rewriting, document relevance grading, hallucination grading, and answer grading continue to operate from the normal raw retrieved documents.
