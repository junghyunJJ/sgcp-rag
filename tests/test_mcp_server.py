import json

import pytest

import mcpserver.mcp_server as mcp_mod

pytestmark = pytest.mark.asyncio


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
    out = await mcp_mod.search_documents("col", "qry")
    assert out == "No results found."


async def test_search_documents_with_results(monkeypatch):
    sample = [
        {"page_content": "Hello", "metadata": {"k": "v"}, "score": 0.5, "id": "doc1"}
    ]

    async def dummy_request(method, endpoint, **kwargs):
        return sample

    monkeypatch.setattr(mcp_mod.client, "request", dummy_request)
    xml = await mcp_mod.search_documents("col", "qry", limit=1, search_type="semantic")
    assert '<search_results type="semantic">' in xml
    assert "<content>Hello</content>" in xml
    assert "<score>0.5000</score>" in xml
    assert "<id>doc1</id>" in xml


async def test_search_documents_bad_filter(monkeypatch):
    out = await mcp_mod.search_documents("col", "qry", filter_json="notjson")
    assert "Error: Invalid JSON in filter parameter" in out


async def test_list_collections_empty(monkeypatch):
    async def dummy_request(method, endpoint, **kwargs):
        return []

    monkeypatch.setattr(mcp_mod.client, "request", dummy_request)
    out = await mcp_mod.list_collections()
    assert out == "No collections found."


async def test_list_collections(monkeypatch):
    data = [{"name": "col1", "uuid": "id1"}]

    async def dummy_request(method, endpoint, **kwargs):
        return data

    monkeypatch.setattr(mcp_mod.client, "request", dummy_request)
    out = await mcp_mod.list_collections()
    assert "- **col1** (ID: id1)" in out


async def test_get_collection(monkeypatch):
    col = {"name": "test", "uuid": "uid"}

    async def dummy_request(method, endpoint, **kwargs):
        return col

    monkeypatch.setattr(mcp_mod.client, "request", dummy_request)
    out = await mcp_mod.get_collection("uid")
    assert "**test**" in out
    assert "ID: uid" in out


async def test_create_collection_invalid_json():
    out = await mcp_mod.create_collection("name", metadata_json="bad")
    assert "Error: Invalid JSON in metadata" in out


async def test_delete_collection(monkeypatch):
    async def dummy_request(method, endpoint, **kwargs):
        return {}

    monkeypatch.setattr(mcp_mod.client, "request", dummy_request)
    out = await mcp_mod.delete_collection("colid")
    assert out == "Collection colid deleted successfully!"


async def test_list_documents_empty(monkeypatch):
    async def dummy_request(method, endpoint, **kwargs):
        return []

    monkeypatch.setattr(mcp_mod.client, "request", dummy_request)
    out = await mcp_mod.list_documents("cid")
    assert out == "No documents found."


async def test_list_documents_with_items(monkeypatch):
    docs = [{"page_content": "x" * 210, "id": "d1"}]

    async def dummy_request(method, endpoint, **kwargs):
        return docs

    monkeypatch.setattr(mcp_mod.client, "request", dummy_request)
    out = await mcp_mod.list_documents("cid", limit=1)
    assert "1." in out
    assert "ID: d1" in out
    assert "..." in out


async def test_add_documents_success(monkeypatch):
    from conftest import DummyAsyncClient

    dummy = DummyAsyncClient({"success": True, "added_chunk_ids": [1, 2, 3]})
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "token")
    monkeypatch.setattr(
        mcp_mod, "client", mcp_mod.LangConnectClient(mcp_mod.API_BASE_URL, "token")
    )
    monkeypatch.setattr(mcp_mod.httpx, "AsyncClient", lambda *args, **kwargs: dummy)
    out = await mcp_mod.add_documents("cid", "text body")
    assert out == "Document added successfully! Created 3 chunks."


async def test_add_documents_failure(monkeypatch):
    from conftest import DummyAsyncClient

    dummy = DummyAsyncClient({"success": False, "message": "err msg"})
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "token")
    monkeypatch.setattr(mcp_mod.httpx, "AsyncClient", lambda *args, **kwargs: dummy)
    out = await mcp_mod.add_documents("cid", "text")
    assert "Failed to add document: err msg" in out


async def test_delete_document(monkeypatch):
    async def dummy_request(method, endpoint, **kwargs):
        return {}

    monkeypatch.setattr(mcp_mod.client, "request", dummy_request)
    out = await mcp_mod.delete_document("cid", "docid")
    assert out == "Document docid deleted successfully!"


async def test_get_health_status(monkeypatch):
    async def dummy_request(method, endpoint, **kwargs):
        return {"status": "ok"}

    monkeypatch.setattr(mcp_mod.client, "request", dummy_request)
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "")
    monkeypatch.setattr(mcp_mod, "SUPABASE_JWT_SECRET", "")
    out = await mcp_mod.get_health_status()
    assert "Status: ok" in out
    assert "Auth: ✗" in out


async def test_multi_query_no_key():
    # Without OPENAI_API_KEY
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(mcp_mod, "OPENAI_API_KEY", "")
    out = await mcp_mod.multi_query("ask?")
    data = json.loads(out)
    assert "error" in data
    monkeypatch.undo()
