#!/usr/bin/env python3
"""LangConnect MCP Server using FastMCP (SSE transport)"""

import json
import os
import sys
from datetime import datetime
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP

from langconnect.agent import query_expansion

LineListOutputParser = query_expansion.LineListOutputParser
generate_query_expansions = query_expansion.generate_query_expansions

if os.getenv("PYTHON_DOTENV_DISABLED") != "1":
    load_dotenv()


# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")
SSE_PORT = int(os.getenv("SSE_PORT", "8765"))


# Create FastMCP server
mcp = FastMCP(name="LangConnect")


# HTTP client
class LangConnectClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def request(self, method: str, endpoint: str, timeout: float = 60.0, **kwargs):
        async with httpx.AsyncClient() as client:
            url = f"{self.base_url}{endpoint}"
            response = await client.request(
                method, url, headers=self.headers, timeout=timeout, **kwargs
            )
            response.raise_for_status()
            return (
                response.json()
                if response.status_code != 204
                else {"status": "success"}
            )


# Initialize client
client = LangConnectClient(API_BASE_URL)


def _delete_partial_success_message(
    exc: httpx.HTTPStatusError,
    collection_id: str,
) -> str | None:
    try:
        detail = exc.response.json().get("detail", {})
    except ValueError:
        return None
    if not isinstance(detail, dict):
        return None
    if detail.get("error") != "documents_deleted_wiki_rebuild_failed":
        return None
    deleted_count = detail.get("deleted_count", "unknown")
    error_id = detail.get("error_id", "unknown")
    return (
        "Document deletion succeeded, but LLM Wiki rebuild failed. "
        f"Deleted count: {deleted_count}. Error ID: {error_id}. "
        f"Run rebuild_llm_wiki({collection_id!r}) to retry."
    )


@mcp.tool
async def search_documents(
    collection_id: str,
    query: str,
    limit: int = 5,
    search_type: str = "semantic",
    filter_json: Optional[str] = None,
) -> str:
    """Search documents in a collection using semantic, keyword, or hybrid search."""
    search_data = {"query": query, "limit": limit, "search_type": search_type}

    if filter_json:
        try:
            search_data["filter"] = json.loads(filter_json)
        except json.JSONDecodeError:
            return "Error: Invalid JSON in filter parameter"

    try:
        results = await client.request(
            "POST", f"/collections/{collection_id}/documents/search",
            timeout=120.0, json=search_data,
        )
    except httpx.ReadTimeout:
        return json.dumps({
            "error": "Search timed out. The query may be too complex or the collection is very large. Try reducing the limit or using a simpler search type.",
            "results": [],
            "count": 0,
        })

    if not results:
        return "No results found."

    output = f"## Search Results ({search_type})\n\n"
    for i, result in enumerate(results, 1):
        output += f"### Result {i} (Score: {result.get('score', 0):.4f})\n"
        output += f"{result.get('page_content', '')}\n"
        output += f"Document ID: {result.get('id', 'Unknown')}\n\n"

    return output


@mcp.tool
async def list_collections() -> str:
    """List all available document collections."""
    collections = await client.request("GET", "/collections")

    if not collections:
        return "No collections found."

    output = "## Collections\n\n"
    for coll in collections:
        output += (
            f"- **{coll.get('name', 'Unnamed')}** (ID: {coll.get('uuid', 'Unknown')})\n"
        )

    return output


@mcp.tool
async def get_collection(collection_id: str) -> str:
    """Get details of a specific collection."""
    collection = await client.request("GET", f"/collections/{collection_id}")
    return f"**{collection.get('name', 'Unnamed')}**\nID: {collection.get('uuid', 'Unknown')}"


@mcp.tool
async def create_collection(name: str, metadata_json: Optional[str] = None) -> str:
    """Create a new collection."""
    data = {"name": name}

    if metadata_json:
        try:
            data["metadata"] = json.loads(metadata_json)
        except json.JSONDecodeError:
            return "Error: Invalid JSON in metadata"

    result = await client.request("POST", "/collections", json=data)
    return f"Collection '{result.get('name')}' created with ID: {result.get('uuid')}"


@mcp.tool
async def delete_collection(collection_id: str) -> str:
    """Delete a collection and all its documents."""
    await client.request("DELETE", f"/collections/{collection_id}")
    return f"Collection {collection_id} deleted successfully!"


@mcp.tool
async def list_documents(collection_id: str, limit: int = 20) -> str:
    """List documents in a collection."""
    docs = await client.request(
        "GET", f"/collections/{collection_id}/documents", params={"limit": limit}
    )

    if not docs:
        return "No documents found."

    output = f"## Documents ({len(docs)} items)\n\n"
    for i, doc in enumerate(docs, 1):
        content_preview = doc.get("page_content", "")[:200]
        if len(doc.get("page_content", "")) > 200:
            content_preview += "..."
        output += f"{i}. {content_preview}\n   ID: {doc.get('id', 'Unknown')}\n\n"

    return output


@mcp.tool
async def add_documents(collection_id: str, text: str) -> str:
    """Add a text document to a collection."""
    metadata = {"source": "mcp-input", "created_at": datetime.now().isoformat()}

    files = [("files", ("document.txt", text.encode("utf-8"), "text/plain"))]
    data = {"metadatas_json": json.dumps([metadata])}

    # Remove Content-Type for multipart
    headers = client.headers.copy()
    headers.pop("Content-Type", None)

    try:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(
                f"{client.base_url}/collections/{collection_id}/documents",
                headers=headers,
                files=files,
                data=data,
                timeout=120.0,
            )
            response.raise_for_status()
            result = response.json()
    except httpx.ReadTimeout:
        return "Error: Document upload timed out. The file may be too large or the server is under heavy load. Try a smaller document or increase chunk_size to reduce the number of chunks."

    if result.get("success"):
        return f"Document added successfully! Created {len(result.get('added_chunk_ids', []))} chunks."
    return f"Failed to add document: {result.get('message', 'Unknown error')}"


@mcp.tool
async def delete_document(collection_id: str, document_id: str) -> str:
    """Delete a document from a collection."""
    try:
        await client.request(
            "DELETE", f"/collections/{collection_id}/documents/{document_id}"
        )
    except httpx.HTTPStatusError as exc:
        message = _delete_partial_success_message(exc, collection_id)
        if message:
            return message
        raise
    return f"Document {document_id} deleted successfully!"


@mcp.tool
async def multi_query(question: str) -> str:
    """Generate multiple queries (3-5) for better vector search results from a single user question."""
    try:
        queries = await generate_query_expansions(question)
        return json.dumps(queries, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Failed to generate queries: {e!s}"})


@mcp.tool
async def agentic_search(
    collection_id: str,
    question: str,
    search_type: str = "hybrid",
    search_limit: int = 5,
    max_rewrites: int = 3,
    filter_json: Optional[str] = None,
    use_wiki_context: bool = True,
) -> str:
    """Run an agentic RAG search that evaluates, rewrites queries, and validates answers.

    Uses an AI agent that retrieves documents, grades relevance, generates an answer,
    and validates it. Automatically retries with rewritten queries if quality is poor.

    Args:
        collection_id: Collection UUID to search in.
        question: The question to answer.
        search_type: "semantic", "keyword", or "hybrid" (default).
        search_limit: Max documents per retrieval. Default 5.
        max_rewrites: Max query rewrite attempts. Default 3.
        filter_json: Optional JSON metadata filter string.
        use_wiki_context: Use existing non-authoritative LLM Wiki navigation context during generation.
    """
    search_data = {
        "question": question,
        "search_type": search_type,
        "search_limit": search_limit,
        "max_rewrites": max_rewrites,
        "use_wiki_context": use_wiki_context,
    }

    if filter_json:
        try:
            search_data["filter"] = json.loads(filter_json)
        except json.JSONDecodeError:
            return json.dumps({
                "error": "Invalid JSON in filter parameter",
                "selected_wiki_pages": [],
                "wiki_context_status": "disabled",
            })

    try:
        result = await client.request(
            "POST",
            f"/collections/{collection_id}/agentic-search",
            timeout=300.0,
            json=search_data,
        )

        output = {
            "answer": result.get("generation", ""),
            "sources": [
                {
                    "content": doc.get("page_content", "")[:300],
                    "metadata": doc.get("metadata", {}),
                    "score": doc.get("score", 0),
                }
                for doc in result.get("relevant_documents", [])
            ],
            "steps": result.get("steps", []),
            "rewrites": result.get("query_rewrites", []),
            "rewrite_count": result.get("rewrite_count", 0),
            "selected_wiki_pages": result.get("selected_wiki_pages", []),
            "wiki_context_status": result.get("wiki_context_status") or "disabled",
        }

        if result.get("error"):
            output["error"] = result["error"]

        return json.dumps(output, ensure_ascii=False)

    except httpx.ReadTimeout:
        return json.dumps({
            "error": "Agentic search timed out. The AI reasoning loop may need more time. Try reducing max_rewrites or using a simpler search_type.",
            "answer": "",
            "sources": [],
            "selected_wiki_pages": [],
            "wiki_context_status": "disabled",
        })
    except Exception as e:
        return json.dumps({
            "error": f"Agentic search failed: {e!s}",
            "selected_wiki_pages": [],
            "wiki_context_status": "disabled",
        })


@mcp.tool
async def rebuild_llm_wiki(
    collection_id: str,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_temperature: Optional[float] = None,
) -> str:
    """Rebuild generated LLM Wiki artifacts for a collection through REST."""
    payload = {
        key: value
        for key, value in {
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "llm_temperature": llm_temperature,
        }.items()
        if value is not None
    }
    result = await client.request(
        "POST",
        f"/collections/{collection_id}/llm-wiki/rebuild",
        timeout=300.0,
        json=payload,
    )
    return json.dumps(result, ensure_ascii=False)


@mcp.tool
async def get_health_status() -> str:
    """Check API health status."""
    result = await client.request("GET", "/health")
    return f"Status: {result.get('status', 'Unknown')}\nAPI: {API_BASE_URL}"


if __name__ == "__main__":
    print("LangConnect MCP SSE Server")
    print("=" * 50)
    print(f"Starting MCP SSE server on http://127.0.0.1:{SSE_PORT}")

    try:
        mcp.run(transport="sse", port=SSE_PORT)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
        sys.exit(1)
