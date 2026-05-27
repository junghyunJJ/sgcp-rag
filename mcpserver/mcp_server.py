#!/usr/bin/env python3
"""LangConnect MCP Server using FastMCP (stdio)"""

import json
import mimetypes
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP

from langconnect.agent import query_expansion
from langconnect.services.paper_cards import repo_relative_path, resolve_repo_root

LineListOutputParser = query_expansion.LineListOutputParser
generate_query_expansions = query_expansion.generate_query_expansions

if os.getenv("PYTHON_DOTENV_DISABLED") != "1":
    load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")

# Create FastMCP server
mcp = FastMCP(
    name="langconnect-rag-mcp",
    instructions="This server provides vector search tools that can be used to search for documents in a collection. Call list_collections() to get a list of available collections. Call get_collection(collection_id) to get details of a specific collection. Call search_documents(collection_id, query, limit, search_type, filter_json) to search for documents in a collection. Call agentic_search(collection_id, question) for AI-powered question answering with automatic query rewriting and answer validation. Call rebuild_llm_wiki(collection_id) to rebuild generated non-authoritative LLM Wiki navigation context for a collection. Call list_documents(collection_id, limit) to list documents in a collection. Call add_documents(collection_id, text) to add a text document to a collection. Call add_documents_from_files(collection_id, file_paths, chunk_size, chunk_overlap) to upload files directly from filesystem (more efficient for large/binary files). Call delete_document(collection_id, document_id) to delete a document from a collection. Call get_health_status() to check the health status of the server.",
)


# Basic dynamic resource returning a string
@mcp.resource("resource://how-to-use-langconnect-rag-mcp")
def get_instructions() -> str:
    """Provides instructions on how to use the LangConnect RAG MCP server."""
    return """
Two approaches for answering questions from documents:

**Option A: Agentic Search (Recommended for Q&A)**
1. Use `list_collections` to find the right **Collection ID**.
2. Use `agentic_search(collection_id, question)` — it automatically retrieves, evaluates, rewrites queries, and validates the answer.

**Option B: Manual Multi-Query Search (For detailed control)**
1. Use `list_collections` to find the right **Collection ID**.
2. Use `multi_query` to generate at least 3 sub-questions.
3. Search all queries with `search_documents` and find useful documents.
4. Use searched documents to answer the question."""


@mcp.prompt("rag-prompt")
async def rag_prompt(query: str) -> list[dict]:
    """Provides a prompt for summarizing the provided text."""
    return [
        {
            "role": "system",
            "content": """You are a question-answer assistant based on given document.
You must use search tool to answer the question.

#Search Configuration:
- Target Collection: (user's request)
- Search Type: hybrid(preferred)
- Search Limit: 5(default)

#Search Guidelines:
Two approaches are available:

**Option A: Agentic Search (Recommended for Q&A)**
1. Use `list_collections` to find the right **Collection ID** for user's request.
2. Use `agentic_search(collection_id, question)` — it automatically retrieves relevant documents, evaluates their quality, rewrites queries if needed, and validates the generated answer.

**Option B: Manual Multi-Query Search (For detailed control)**
1. Use `list_collections` to find the right **Collection ID** for user's request.
2. Use `multi_query` to generate at least 3 sub-questions which are related to original user's request.
3. Search all queries generated from previous step(`multi_query`) and find useful documents from collection.
4. Use searched documents to answer the question.

---

## Format:
(answer to the question)

**Source**
- [1] (Source and page numbers)
- [2] (Source and page numbers)
- ...

---

[Note]
- Answer in same language as user's request
- Append sources that you've referenced at the very end of your answer.
- If you can't find your answer from <search_results>, just say you can't find any relevant source to answer the question without any narrative sentences.
""",
        },
        {"role": "user", "content": f"User's request:\n\n{query}"},
    ]


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


# Helper function for file path validation
def validate_file_path(file_path: str) -> Path:
    """Validate and resolve file path safely.

    Args:
        file_path: Path to the file to validate

    Returns:
        Resolved Path object

    Raises:
        ValueError: If file doesn't exist, is not a file, or other validation errors
    """
    try:
        path = Path(file_path).expanduser().resolve()
    except Exception as e:
        raise ValueError(f"Invalid file path '{file_path}': {e}")

    if not path.exists():
        raise ValueError(f"File not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Not a file (possibly a directory): {file_path}")

    return path


@mcp.tool
async def search_documents(
    collection_id: str,
    query: str,
    limit: int = 5,
    search_type: str = "semantic",
    filter_json: Optional[str] = None,
) -> str:
    """Search documents in a collection using semantic, keyword, or hybrid search.

    This function is used to find relevant documents within a specific collection based on a search query.
    It supports multiple search types to provide flexible document retrieval capabilities.
    The function returns structured search results with document content, metadata, relevance scores, and document IDs.

    Args:
        collection_id: The unique identifier of the collection to search in. This should be obtained
                      from the list_collections() function or provided by the user.
        query: The search query string to find relevant documents. This can be a natural language
               question, keywords, or any text that describes what you're looking for.
        limit: Maximum number of documents to return. Default is 5, maximum allowed is 100.
               Higher limits provide more results but may take longer to process.
        search_type: Type of search algorithm to perform. Options include:
                    - "semantic": Uses vector similarity search (recommended for natural language queries)
                    - "keyword": Uses traditional text matching (good for exact terms)
                    - "hybrid": Combines both semantic and keyword search (best overall results)
        filter_json: Optional JSON string containing metadata filters to narrow down the search scope.
                    Example: '{"source": "sample.pdf", "category": "technical"}'
                    This helps focus the search on specific document types or sources.
    """
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
        return json.dumps({"results": [], "count": 0, "search_type": search_type})

    # Return JSON format for easier programmatic parsing
    output = {
        "results": [
            {
                "content": result.get("page_content", ""),
                "metadata": result.get("metadata", {}),
                "score": round(result.get("score", 0), 4),
                "id": result.get("id", "Unknown"),
            }
            for result in results
        ],
        "count": len(results),
        "search_type": search_type,
    }

    return json.dumps(output, ensure_ascii=False)


@mcp.tool
async def list_collections() -> str:
    """List all available document collections.

    This function retrieves and displays all document collections that are available in the system.
    It's typically the first step in the RAG workflow to identify which collection contains
    the relevant documents for a user's query. The function returns structured information
    about each collection including names, IDs, and metadata. Use this function to discover
    what collections are available before performing searches or other operations.

    Returns:
        str: JSON string containing a list of collections with their names, IDs, and metadata.
              Format: {"collections": [{"name": "...", "id": "...", "metadata": {...}}], "count": N}
              If no collections are found, returns a message indicating this.
    """
    collections = await client.request("GET", "/collections")

    if not collections:
        return '{"collections": [], "message": "No collections found."}'

    # Format collections as structured data
    formatted_collections = []
    for coll in collections:
        formatted_collections.append(
            {
                "name": coll.get("name", "Unnamed"),
                "id": coll.get("uuid", "Unknown"),
                "metadata": coll.get("metadata", {}),
            }
        )

    return json.dumps(
        {"collections": formatted_collections, "count": len(formatted_collections)},
        indent=2,
    )


@mcp.tool
async def get_collection(collection_id: str) -> str:
    """Get details of a specific collection.

    This function retrieves detailed information about a specific document collection.
    It's useful for verifying collection details, checking metadata, or confirming
    that a collection exists before performing operations on it. The function provides
    basic information about the collection including its name and unique identifier.

    Args:
        collection_id: The unique identifier of the collection to retrieve. This should be
                      obtained from the list_collections() function or provided by the user.
                      Must be a valid UUID string for an existing collection.

    Returns:
        str: Formatted string containing the collection name and ID in a readable format.
             Format: "**Collection Name**\nID: collection-uuid"
    """
    collection = await client.request("GET", f"/collections/{collection_id}")
    return f"**{collection.get('name', 'Unnamed')}**\nID: {collection.get('uuid', 'Unknown')}"


@mcp.tool
async def create_collection(name: str, metadata_json: Optional[str] = None) -> str:
    """Create a new collection.

    This function creates a new document collection in the system. Collections are containers
    that hold related documents and enable organized storage and retrieval of information.
    Each collection can have custom metadata to provide additional context and categorization.
    Once created, documents can be added to the collection using add_documents() function.

    Args:
        name: The name of the collection to create. Should be descriptive and help identify
              the purpose or content of the collection. Must be a non-empty string.
        metadata_json: Optional JSON string containing metadata for the collection.
                      This can include additional information such as description, tags,
                      creation context, or any other relevant details.
                      Example: '{"description": "My collection", "tags": ["tag1", "tag2"]}'
                      If provided, must be valid JSON format.

    Returns:
        str: Success message with the created collection name and ID.
             Format: "Collection 'name' created with ID: collection-uuid"
             If JSON parsing fails, returns an error message.
    """
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
    """Delete a collection and all its documents.

    This function permanently removes a collection and all documents contained within it
    from the system. This is a destructive operation that cannot be undone, so it should
    be used with caution. All document chunks, metadata, and embeddings associated with
    the collection will be permanently deleted. Use this function only when you are certain
    that the collection is no longer needed.

    Args:
        collection_id: The unique identifier of the collection to delete. This should be
                      obtained from the list_collections() function or provided by the user.
                      Must be a valid UUID string for an existing collection.

    Returns:
        str: Success message confirming the collection deletion.
             Format: "Collection collection-uuid deleted successfully!"
    """
    await client.request("DELETE", f"/collections/{collection_id}")
    return f"Collection {collection_id} deleted successfully!"


@mcp.tool
async def list_documents(collection_id: str, limit: int = 20) -> str:
    """List documents in a collection.

    This function retrieves and displays all documents stored in a specific collection.
    It provides a paginated view of documents with their content previews and unique
    identifiers. This is useful for exploring the contents of a collection, verifying
    document uploads, or finding specific document IDs for operations like deletion.
    The function shows a preview of each document's content (first 200 characters) to
    help identify documents without retrieving full content.

    Args:
        collection_id: The unique identifier of the collection to list documents from.
                      This should be obtained from the list_collections() function or
                      provided by the user. Must be a valid UUID string for an existing collection.
        limit: Maximum number of documents to return. Default is 20, which helps manage
               large collections by providing pagination. Higher values will return more
               documents but may take longer to process and display.

    Returns:
        str: Formatted string containing a numbered list of documents with content previews
             and IDs. Format: "## Documents (N items)\n\n1. [content preview...]\n   ID: doc-id"
             If no documents are found, returns "No documents found."
    """
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
async def add_documents(
    collection_id: str,
    text: str,
    chunk_size: int = 3000,
    chunk_overlap: int = 200,
    filename: str = "document.txt",
) -> str:
    """Add a text document to a collection.

    This function adds a new text document to an existing collection. The document text
    will be processed, chunked into smaller segments for optimal vector search performance,
    and stored with embeddings for semantic search capabilities. Each document is automatically
    tagged with metadata including source information and creation timestamp. The function
    supports adding plain text content and will handle the chunking and embedding process
    automatically.

    Args:
        collection_id: The unique identifier of the collection to add the document to.
                      This should be obtained from the list_collections() function or
                      provided by the user. Must be a valid UUID string for an existing collection.
        text: The text content of the document to add. This should be the full text content
              that you want to make searchable. The text will be automatically chunked into
              smaller segments for optimal retrieval performance. Can be any length, but
              very large texts will be processed in chunks.
        chunk_size: Maximum number of characters in each chunk (default: 3000).
                   Larger chunks preserve more context but may reduce precision.
        chunk_overlap: Number of overlapping characters between chunks (default: 200).
                      Overlap helps maintain context across chunk boundaries.
        filename: Optional filename for the document (default: "document.txt").
                 Used for tracking and metadata purposes.

    Returns:
        str: Success message indicating the document was added and the number of chunks created.
             Format: "Document added successfully! Created N chunks."
             If the operation fails, returns an error message with details.
    """
    # Validate text size (10MB limit for safety)
    text_bytes = text.encode("utf-8")
    max_size = 10 * 1024 * 1024  # 10MB
    if len(text_bytes) > max_size:
        return f"Error: Document too large ({len(text_bytes)} bytes). Maximum size is {max_size} bytes (10MB)."

    metadata = {
        "source": "mcp-input",
        "created_at": datetime.now().isoformat(),
        "filename": filename,
    }

    files = [("files", (filename, text_bytes, "text/plain"))]
    data = {
        "metadatas_json": json.dumps([metadata]),
        "chunk_size": str(chunk_size),
        "chunk_overlap": str(chunk_overlap),
    }

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
        chunks_created = len(result.get("added_chunk_ids", []))
        message = f"Document added successfully! Created {chunks_created} chunks from {len(text_bytes)} bytes."
        if result.get("warnings"):
            message += f"\nWarnings: {result['warnings']}"
        return message
    return f"Failed to add document: {result.get('message', 'Unknown error')}"


@mcp.tool
async def add_documents_from_files(
    collection_id: str,
    file_paths: list[str] | str,
    chunk_size: int = 3000,
    chunk_overlap: int = 200,
) -> str:
    """Upload files directly from local filesystem without reading content into text first.

    This function is more efficient than add_documents() for large files or binary files
    (PDF, DOCX, etc.) because it doesn't require loading file content into Claude's context.
    The MCP server reads the files directly and uploads them to the API. This approach
    saves context window space and supports all file types that the API can process.

    Supported file types: PDF, DOCX, TXT, MD, HTML, and any other text-based formats.

    Args:
        collection_id: The unique identifier of the collection to add documents to.
                      This should be obtained from the list_collections() function or
                      provided by the user. Must be a valid UUID string for an existing collection.
        file_paths: List of absolute or relative file paths to upload, OR a single file path string.
                   Each path will be validated before processing. Can upload multiple files in a
                   single call for batch processing efficiency.
                   Example: ["/path/to/doc1.pdf", "~/doc2.txt"] or "/path/to/single.pdf"
        chunk_size: Maximum number of characters in each chunk (default: 3000).
                   Larger chunks preserve more context but may reduce precision.
        chunk_overlap: Number of overlapping characters between chunks (default: 200).
                      Overlap helps maintain context across chunk boundaries.

    Returns:
        str: Success message indicating the number of files processed and chunks created.
             Format: "Uploaded N file(s) successfully! Created M total chunks."
             If some files fail, includes warnings about which files failed.
             If all files fail, returns error message with details.

    Examples:
        >>> add_documents_from_files("uuid-123", ["/path/to/document.pdf"])
        "Uploaded 1 file(s) successfully! Created 45 total chunks."

        >>> add_documents_from_files("uuid-123", ["~/paper1.pdf", "~/paper2.txt"], chunk_size=500)
        "Uploaded 2 file(s) successfully! Created 128 total chunks."

        >>> add_documents_from_files("uuid-123", "/path/to/single.txt")
        "Uploaded 1 file(s) successfully! Created 12 total chunks."
    """
    # CRITICAL FIX: Handle parameter serialization from MCP
    # Claude Desktop may pass file_paths as string instead of list
    if isinstance(file_paths, str):
        print(f"[MCP DEBUG] Received file_paths as string: {file_paths}", file=sys.stderr, flush=True)
        # Try to parse as JSON array
        try:
            parsed = json.loads(file_paths)
            if isinstance(parsed, list):
                file_paths = parsed
                print(f"[MCP DEBUG] Parsed JSON to list: {file_paths}", file=sys.stderr, flush=True)
            else:
                # Single file path as string (not JSON)
                file_paths = [file_paths]
                print("[MCP DEBUG] Wrapped single path in list", file=sys.stderr, flush=True)
        except json.JSONDecodeError:
            # Single file path as plain string
            file_paths = [file_paths]
            print("[MCP DEBUG] Not JSON, wrapped in list", file=sys.stderr, flush=True)

    if not isinstance(file_paths, list):
        error_msg = f"Error: file_paths must be a list or string, got {type(file_paths).__name__}"
        print(f"[MCP ERROR] {error_msg}", file=sys.stderr, flush=True)
        return error_msg

    print(f"[MCP DEBUG] Processing {len(file_paths)} file(s)", file=sys.stderr, flush=True)
    # Validate all file paths first
    validated_files: list[tuple[Path, str]] = []
    failed_files: list[str] = []

    for file_path in file_paths:
        try:
            path = validate_file_path(file_path)

            # Check file size (10MB limit)
            file_size = path.stat().st_size
            max_size = 10 * 1024 * 1024  # 10MB
            if file_size > max_size:
                failed_files.append(f"{file_path} (too large: {file_size} bytes)")
                continue

            # Detect MIME type
            mime_type, _ = mimetypes.guess_type(str(path))
            if not mime_type:
                mime_type = "application/octet-stream"

            validated_files.append((path, mime_type))

        except ValueError as e:
            failed_files.append(f"{file_path} ({str(e)})")

    if not validated_files:
        error_msg = "Failed to validate any files."
        if failed_files:
            error_msg += "\nFailed files:\n" + "\n".join(f"  - {f}" for f in failed_files)
        return error_msg

    # Prepare files for upload
    # FIX: Use synchronous file reading instead of asyncio.to_thread()
    # This is more reliable in MCP stdio context and files are small (< 10MB)
    files_to_upload = []
    metadatas = []
    repo_root = resolve_repo_root()

    for path, mime_type in validated_files:
        try:
            # Read file content synchronously
            # This is safe because:
            # 1. Files are limited to 10MB
            # 2. Modern SSDs make small file I/O very fast
            # 3. Avoids thread pool executor issues in MCP context
            file_content = path.read_bytes()

            # Pass bytes content to httpx (not file handle)
            files_to_upload.append(("files", (path.name, file_content, mime_type)))

            # Create metadata for each file
            metadata = {
                "source": path.name,  # Use original filename as source
                "created_at": datetime.now().isoformat(),
                "filename": path.name,
                "mime_type": mime_type,
            }
            source_path = repo_relative_path(path, repo_root)
            if source_path:
                metadata["source_path"] = source_path
            metadatas.append(metadata)
        except Exception as e:
            print(f"[MCP ERROR] Failed to read {path}: {e}", file=sys.stderr, flush=True)
            failed_files.append(f"{path.name} (read error: {str(e)})")

    if not files_to_upload:
        return "Failed to read any files for upload."

    # Prepare form data
    data = {
        "metadatas_json": json.dumps(metadatas),
        "chunk_size": str(chunk_size),
        "chunk_overlap": str(chunk_overlap),
    }

    # Remove Content-Type for multipart
    headers = client.headers.copy()
    headers.pop("Content-Type", None)

    # Upload to API with bytes content (no file handles to manage)
    try:
        print(f"[MCP DEBUG] Uploading {len(files_to_upload)} file(s) to API...", file=sys.stderr, flush=True)
        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(
                f"{client.base_url}/collections/{collection_id}/documents",
                headers=headers,
                files=files_to_upload,
                data=data,
                timeout=120.0,
            )
            print(f"[MCP DEBUG] Response status: {response.status_code}", file=sys.stderr, flush=True)
            response.raise_for_status()
            result = response.json()

        if result.get("success"):
            chunks_created = len(result.get("added_chunk_ids", []))
            message = (
                f"Uploaded {len(validated_files)} file(s) successfully! "
                f"Created {chunks_created} total chunks."
            )

            if failed_files:
                message += "\n\nWarning - Failed files:\n" + "\n".join(f"  - {f}" for f in failed_files)

            if result.get("warnings"):
                message += f"\n\nAPI warnings: {result['warnings']}"

            print(f"[MCP DEBUG] SUCCESS: {message}", file=sys.stderr, flush=True)
            return message
        else:
            error_msg = f"Failed to upload documents: {result.get('message', 'Unknown error')}"
            print(f"[MCP ERROR] {error_msg}", file=sys.stderr, flush=True)
            return error_msg

    except httpx.ReadTimeout:
        error_msg = "File upload timed out. The files may be too large or the server is under heavy load. Try uploading fewer files or increasing chunk_size."
        print(f"[MCP ERROR] {error_msg}", file=sys.stderr, flush=True)
        return error_msg
    except httpx.HTTPError as e:
        error_msg = f"HTTP error during upload: {str(e)}"
        print(f"[MCP ERROR] {error_msg}", file=sys.stderr, flush=True)
        if hasattr(e, "response") and e.response is not None:
            print(f"[MCP ERROR] Response: {e.response.text}", file=sys.stderr, flush=True)
        return error_msg
    except Exception as e:
        error_msg = f"Unexpected error during upload: {str(e)}"
        print(f"[MCP ERROR] {error_msg}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return error_msg


@mcp.tool
async def delete_document(collection_id: str, document_id: str) -> str:
    """Delete a document from a collection.

    This function permanently removes a specific document and all its associated chunks
    from a collection. This is a destructive operation that cannot be undone, so it should
    be used with caution. All chunks, metadata, and embeddings associated with the document
    will be permanently deleted. Use this function to remove outdated, incorrect, or
    no longer needed documents from a collection.

    Args:
        collection_id: The unique identifier of the collection containing the document.
                      This should be obtained from the list_collections() function or
                      provided by the user. Must be a valid UUID string for an existing collection.
        document_id: The unique identifier of the document to delete. This should be
                    obtained from the list_documents() function or search results.
                    Must be a valid document ID that exists in the specified collection.

    Returns:
        str: Success message confirming the document deletion.
             Format: "Document document-id deleted successfully!"
    """
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
async def agentic_search(
    collection_id: str,
    question: str,
    search_type: str = "hybrid",
    search_limit: int = 5,
    max_rewrites: int = 3,
    filter_json: Optional[str] = None,
    use_wiki_context: bool = True,
) -> str:
    """Run an agentic RAG search that automatically evaluates, rewrites queries, and validates answers.

    Unlike search_documents which returns raw document chunks, this function uses an AI agent
    that retrieves documents, grades their relevance, generates an answer, and validates it.
    If the retrieval or answer quality is poor, the agent automatically rewrites the query
    and retries (up to max_rewrites times). This is the recommended search method for
    question-answering tasks where you need a direct answer rather than raw documents.

    Args:
        collection_id: The unique identifier of the collection to search in.
        question: The question to answer. Should be a clear, well-formed question.
        search_type: Search algorithm: "semantic", "keyword", or "hybrid" (default).
        search_limit: Maximum documents per retrieval attempt. Default is 5.
        max_rewrites: Maximum number of query rewrite attempts. Default is 3.
        filter_json: Optional JSON string with metadata filters.
                    Example: '{"source": "paper.pdf"}'
        use_wiki_context: Use existing non-authoritative LLM Wiki navigation context during generation.

    Returns:
        str: JSON string with the answer, source documents, execution trace, and any rewrites.
             Format: {"answer": "...", "sources": [...], "steps": [...], "rewrites": [...]}
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
    """Check API health status.

    This function performs a health check on the LangConnect API server to verify
    that it is running and accessible. It also provides information about the current
    configuration, including the API base URL and authentication status. This is useful
    for troubleshooting connection issues, verifying server availability, and confirming
    that the MCP server is properly configured to communicate with the API.

    Returns:
        str: Formatted string containing the health status, API URL, and authentication status.
             Format: "Status: {status}\nAPI: {url}\nAuth: {auth_status}"
             The auth status shows "✓" if authentication is configured, "✗" if not.
    """
    result = await client.request("GET", "/health")
    return f"Status: {result.get('status', 'Unknown')}\nAPI: {API_BASE_URL}"


@mcp.tool
async def multi_query(question: str) -> str:
    """Generate multiple queries (3-5) for better vector search results from a single user question.

    This function uses an LLM to generate multiple variations of a user's question to improve
    vector search results. By creating different perspectives and phrasings of the same question,
    it helps overcome limitations of distance-based similarity search and increases the likelihood
    of finding relevant documents. This is particularly useful for complex queries or when the
    original question might not match the exact wording used in the documents. The generated
    queries can then be used with the search_documents() function to perform comprehensive searches.

    Args:
        question: The original user question to generate variations for. This should be a
                 clear, well-formed question that you want to search for in the document
                 collection. The function will create 3-5 alternative phrasings and
                 perspectives of this question to improve search coverage.

    Returns:
        str: JSON array string containing 3-5 alternative queries generated from the original question.
             Format: ["query1", "query2", "query3", ...]
             If query expansion LLM configuration is invalid, returns an error message.
             If query generation fails, returns an error message with details.
    """
    try:
        queries = await generate_query_expansions(question)
        return json.dumps(queries, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Failed to generate queries: {e!s}"})


def main():
    """Entry point for the MCP server"""
    mcp.run()


if __name__ == "__main__":
    main()
