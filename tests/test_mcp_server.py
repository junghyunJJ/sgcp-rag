import json

import pytest

import mcpserver.mcp_server as mcp_mod

pytestmark = pytest.mark.asyncio


# FastMCP's @mcp.tool wraps functions in FunctionTool objects.
# Access the underlying function via .fn for direct testing.
search_documents = mcp_mod.search_documents.fn
list_collections = mcp_mod.list_collections.fn
get_collection = mcp_mod.get_collection.fn
create_collection = mcp_mod.create_collection.fn
delete_collection = mcp_mod.delete_collection.fn
list_documents = mcp_mod.list_documents.fn
add_documents = mcp_mod.add_documents.fn
add_documents_from_files = mcp_mod.add_documents_from_files.fn
delete_document = mcp_mod.delete_document.fn
agentic_search = mcp_mod.agentic_search.fn
get_health_status = mcp_mod.get_health_status.fn
multi_query = mcp_mod.multi_query.fn


@pytest.mark.parametrize(
    "text,expected",
    [
        ("a\nb\n\n c \n", ["a", "b", "c"]),
        ("\n1\n2\n", ["1", "2"]),
    ],
)
async def test_line_list_parser(text, expected):
    parser = mcp_mod.LineListOutputParser()
    assert parser.parse(text) == expected


async def test_search_documents_no_results(monkeypatch):
    async def dummy_request(method, endpoint, **kwargs):
        return []

    monkeypatch.setattr(mcp_mod.client, "request", dummy_request)
    out = await search_documents("col", "qry")
    data = json.loads(out)
    assert data["results"] == []
    assert data["count"] == 0


async def test_search_documents_with_results(monkeypatch):
    sample = [
        {"page_content": "Hello", "metadata": {"k": "v"}, "score": 0.5, "id": "doc1"}
    ]

    async def dummy_request(method, endpoint, **kwargs):
        return sample

    monkeypatch.setattr(mcp_mod.client, "request", dummy_request)
    out = await search_documents("col", "qry", limit=1, search_type="semantic")
    data = json.loads(out)
    assert data["count"] == 1
    assert data["search_type"] == "semantic"
    assert data["results"][0]["content"] == "Hello"
    assert data["results"][0]["score"] == 0.5
    assert data["results"][0]["id"] == "doc1"


async def test_search_documents_bad_filter(monkeypatch):
    out = await search_documents("col", "qry", filter_json="notjson")
    assert "Error: Invalid JSON in filter parameter" in out


async def test_list_collections_empty(monkeypatch):
    async def dummy_request(method, endpoint, **kwargs):
        return []

    monkeypatch.setattr(mcp_mod.client, "request", dummy_request)
    out = await list_collections()
    data = json.loads(out)
    assert data["collections"] == []


async def test_list_collections(monkeypatch):
    data = [{"name": "col1", "uuid": "id1"}]

    async def dummy_request(method, endpoint, **kwargs):
        return data

    monkeypatch.setattr(mcp_mod.client, "request", dummy_request)
    out = await list_collections()
    parsed = json.loads(out)
    assert parsed["count"] == 1
    assert parsed["collections"][0]["name"] == "col1"
    assert parsed["collections"][0]["id"] == "id1"


async def test_get_collection(monkeypatch):
    col = {"name": "test", "uuid": "uid"}

    async def dummy_request(method, endpoint, **kwargs):
        return col

    monkeypatch.setattr(mcp_mod.client, "request", dummy_request)
    out = await get_collection("uid")
    assert "**test**" in out
    assert "ID: uid" in out


async def test_create_collection_invalid_json():
    out = await create_collection("name", metadata_json="bad")
    assert "Error: Invalid JSON in metadata" in out


async def test_delete_collection(monkeypatch):
    async def dummy_request(method, endpoint, **kwargs):
        return {}

    monkeypatch.setattr(mcp_mod.client, "request", dummy_request)
    out = await delete_collection("colid")
    assert out == "Collection colid deleted successfully!"


async def test_list_documents_empty(monkeypatch):
    async def dummy_request(method, endpoint, **kwargs):
        return []

    monkeypatch.setattr(mcp_mod.client, "request", dummy_request)
    out = await list_documents("cid")
    assert out == "No documents found."


async def test_list_documents_with_items(monkeypatch):
    docs = [{"page_content": "x" * 210, "id": "d1"}]

    async def dummy_request(method, endpoint, **kwargs):
        return docs

    monkeypatch.setattr(mcp_mod.client, "request", dummy_request)
    out = await list_documents("cid", limit=1)
    assert "1." in out
    assert "ID: d1" in out
    assert "..." in out


async def test_add_documents_success(monkeypatch):
    from conftest import DummyAsyncClient

    dummy = DummyAsyncClient({"success": True, "added_chunk_ids": [1, 2, 3]})
    monkeypatch.setattr(
        mcp_mod, "client", mcp_mod.LangConnectClient(mcp_mod.API_BASE_URL)
    )
    monkeypatch.setattr(mcp_mod.httpx, "AsyncClient", lambda *args, **kwargs: dummy)
    out = await add_documents("cid", "text body")
    assert "Document added successfully!" in out
    assert "3 chunks" in out


async def test_add_documents_failure(monkeypatch):
    from conftest import DummyAsyncClient

    dummy = DummyAsyncClient({"success": False, "message": "err msg"})
    monkeypatch.setattr(mcp_mod.httpx, "AsyncClient", lambda *args, **kwargs: dummy)
    out = await add_documents("cid", "text")
    assert "Failed to add document: err msg" in out


async def test_delete_document(monkeypatch):
    async def dummy_request(method, endpoint, **kwargs):
        return {}

    monkeypatch.setattr(mcp_mod.client, "request", dummy_request)
    out = await delete_document("cid", "docid")
    assert out == "Document docid deleted successfully!"


async def test_get_health_status(monkeypatch):
    async def dummy_request(method, endpoint, **kwargs):
        return {"status": "ok"}

    monkeypatch.setattr(mcp_mod.client, "request", dummy_request)
    out = await get_health_status()
    assert "Status: ok" in out
    assert "API:" in out


async def test_multi_query_openai_provider_no_key():
    # Explicit OpenAI query expansion should report missing credentials.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("QUERY_EXPANSION_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    out = await multi_query("ask?")
    data = json.loads(out)
    assert "error" in data
    assert "OpenAI API key" in data["error"]
    monkeypatch.undo()


async def test_multi_query_uses_shared_query_expansion_helper(monkeypatch):
    """stdio MCP multi_query should delegate LLM selection to the shared helper."""
    captured: list[str] = []

    async def fake_generate(question: str) -> list[str]:
        captured.append(question)
        return ["query one", "query two"]

    monkeypatch.setattr(mcp_mod, "generate_query_expansions", fake_generate)

    out = await multi_query("ask?")

    assert json.loads(out) == ["query one", "query two"]
    assert captured == ["ask?"]


async def test_sse_multi_query_uses_shared_query_expansion_helper(monkeypatch):
    """SSE MCP multi_query should use the same query expansion helper."""
    import mcpserver.mcp_sse_server as sse_mod

    captured: list[str] = []

    async def fake_generate(question: str) -> list[str]:
        captured.append(question)
        return ["sse query"]

    monkeypatch.setattr(sse_mod, "generate_query_expansions", fake_generate)

    out = await sse_mod.multi_query.fn("ask?")

    assert json.loads(out) == ["sse query"]
    assert captured == ["ask?"]


async def test_agentic_search_forwards_wiki_context_and_returns_metadata(monkeypatch):
    """Forward wiki context opt-in through the stdio MCP wrapper."""
    captured: dict[str, object] = {}

    async def dummy_request(
        method: str,
        endpoint: str,
        **kwargs: object,
    ) -> dict[str, object]:
        captured.update({"method": method, "endpoint": endpoint, **kwargs})
        return {
            "generation": "answer",
            "relevant_documents": [],
            "steps": ["wiki_context: selected 1 pages"],
            "query_rewrites": [],
            "rewrite_count": 0,
            "selected_wiki_pages": [{"id": "wiki", "title": "Wiki"}],
            "wiki_context_status": "selected",
        }

    monkeypatch.setattr(mcp_mod.client, "request", dummy_request)

    out = await agentic_search("cid", "question?", use_wiki_context=True)
    data = json.loads(out)

    assert captured["json"]["use_wiki_context"] is True
    assert data["selected_wiki_pages"] == [{"id": "wiki", "title": "Wiki"}]
    assert data["wiki_context_status"] == "selected"


async def test_sse_agentic_search_forwards_wiki_context_and_returns_metadata(
    monkeypatch,
):
    """Forward wiki context opt-in through the SSE MCP wrapper."""
    import mcpserver.mcp_sse_server as sse_mod

    captured: dict[str, object] = {}

    async def dummy_request(
        method: str,
        endpoint: str,
        **kwargs: object,
    ) -> dict[str, object]:
        captured.update({"method": method, "endpoint": endpoint, **kwargs})
        return {
            "generation": "answer",
            "relevant_documents": [],
            "steps": ["wiki_context: selected 1 pages"],
            "query_rewrites": [],
            "rewrite_count": 0,
            "selected_wiki_pages": [{"id": "wiki", "title": "Wiki"}],
            "wiki_context_status": "selected",
        }

    monkeypatch.setattr(sse_mod.client, "request", dummy_request)

    out = await sse_mod.agentic_search.fn(
        "cid",
        "question?",
        use_wiki_context=True,
    )
    data = json.loads(out)

    assert captured["json"]["use_wiki_context"] is True
    assert data["selected_wiki_pages"] == [{"id": "wiki", "title": "Wiki"}]
    assert data["wiki_context_status"] == "selected"


async def test_agentic_search_invalid_filter_returns_finite_wiki_status():
    """Invalid filter errors should still include wiki status metadata."""
    out = await agentic_search("cid", "question?", filter_json="not-json")
    data = json.loads(out)

    assert data["error"] == "Invalid JSON in filter parameter"
    assert data["selected_wiki_pages"] == []
    assert data["wiki_context_status"] == "disabled"

    import mcpserver.mcp_sse_server as sse_mod

    sse_out = await sse_mod.agentic_search.fn(
        "cid",
        "question?",
        filter_json="not-json",
    )
    sse_data = json.loads(sse_out)

    assert sse_data["error"] == "Invalid JSON in filter parameter"
    assert sse_data["selected_wiki_pages"] == []
    assert sse_data["wiki_context_status"] == "disabled"
