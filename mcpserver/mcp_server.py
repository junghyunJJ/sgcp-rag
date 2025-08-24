#!/usr/bin/env python3
"""LangConnect MCP Server using FastMCP (stdio)"""

import json
import os
from datetime import datetime
from typing import Optional

import httpx
from dotenv import load_dotenv
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from fastmcp import FastMCP

load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Create FastMCP server
mcp = FastMCP(
    name="langconnect-rag-mcp",
    instructions="This server provides vector search tools that can be used to search for documents in a collection. Call list_collections() to get a list of available collections. Call get_collection(collection_id) to get details of a specific collection. Call search_documents(collection_id, query, limit, search_type, filter_json) to search for documents in a collection. Call list_documents(collection_id, limit) to list documents in a collection. Call add_documents(collection_id, text) to add a text document to a collection. Call delete_document(collection_id, document_id) to delete a document from a collection. Call get_health_status() to check the health status of the server.",
)


# Basic dynamic resource returning a string
@mcp.resource("resource://how-to-use-langconnect-rag-mcp")
def get_instructions() -> str:
    """Provides instructions on how to use the LangConnect RAG MCP server."""
    return """
Follow the guidelines step-by-step to find the answer.
1. Use `list_collections` to list up collections and find right **Collection ID** for user's request.
2. Use `multi_query` to generate at least 3 sub-questions which are related to original user's request.
3. Search all queries generated from previous step(`multi_query`) and find useful documents from collection.
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
Follow the guidelines step-by-step to find the answer.
1. Use `list_collections` to list up collections and find right **Collection ID** for user's request.
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


# Output parser for multi-query generation
class LineListOutputParser(BaseOutputParser[list[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> list[str]:
        # Split into lines, strip whitespace, and remove empties
        lines = [line.strip() for line in text.strip().split("\n")]
        return [line for line in lines if line]


# HTTP client
class LangConnectClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    async def request(self, method: str, endpoint: str, **kwargs):
        async with httpx.AsyncClient() as client:
            url = f"{self.base_url}{endpoint}"
            response = await client.request(
                method, url, headers=self.headers, timeout=60.0, **kwargs
            )
            response.raise_for_status()
            return (
                response.json()
                if response.status_code != 204
                else {"status": "success"}
            )


# Initialize client
client = LangConnectClient(API_BASE_URL, SUPABASE_JWT_SECRET)


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

    results = await client.request(
        "POST", f"/collections/{collection_id}/documents/search", json=search_data
    )

    if not results:
        return "No results found."

    output = f'<search_results type="{search_type}">\n'
    for i, result in enumerate(results, 1):
        output += "  <document>\n"
        output += f"    <content>{result.get('page_content', '')}</content>\n"
        output += f"    <metadata>{json.dumps(result.get('metadata', {}), ensure_ascii=False)}</metadata>\n"
        output += f"    <score>{result.get('score', 0):.4f}</score>\n"
        output += f"    <id>{result.get('id', 'Unknown')}</id>\n"
        output += "  </document>\n"
    output += "</search_results>"

    return output


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
async def add_documents(collection_id: str, text: str) -> str:
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

    Returns:
        str: Success message indicating the document was added and the number of chunks created.
             Format: "Document added successfully! Created N chunks."
             If the operation fails, returns an error message with details.
    """
    metadata = {"source": "mcp-input", "created_at": datetime.now().isoformat()}

    files = [("files", ("document.txt", text.encode("utf-8"), "text/plain"))]
    data = {"metadatas_json": json.dumps([metadata])}

    # Remove Content-Type for multipart
    headers = client.headers.copy()
    headers.pop("Content-Type", None)

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

    if result.get("success"):
        return f"Document added successfully! Created {len(result.get('added_chunk_ids', []))} chunks."
    return f"Failed to add document: {result.get('message', 'Unknown error')}"


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
    await client.request(
        "DELETE", f"/collections/{collection_id}/documents/{document_id}"
    )
    return f"Document {document_id} deleted successfully!"


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
    return f"Status: {result.get('status', 'Unknown')}\nAPI: {API_BASE_URL}\nAuth: {'✓' if SUPABASE_JWT_SECRET else '✗'}"


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
             If Google API key is not configured, returns an error message.
             If query generation fails, returns an error message with details.
    """
    if not GOOGLE_API_KEY:
        return json.dumps({"error": "Google API key not configured"})

    try:
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, api_key=GOOGLE_API_KEY)
        # llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY)

        # Create prompt template
        query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate 3 to 5 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Do not number them.
Original question: {question}""",
        )

        # Create parser
        output_parser = LineListOutputParser()

        # Create chain
        chain = query_prompt | llm | output_parser

        # Generate queries
        queries = await chain.ainvoke({"question": question})

        # Return as JSON array
        return json.dumps(queries, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Failed to generate queries: {e!s}"})


def main():
    """Entry point for the MCP server"""
    import sys

    # print("Starting LangConnect MCP server...", file=sys.stderr)
    # print(f"API_BASE_URL: {API_BASE_URL}", file=sys.stderr)
    # print(
    #     f"SUPABASE_JWT_SECRET configured: {'Yes' if SUPABASE_JWT_SECRET else 'No'}",
    #     file=sys.stderr,
    # )
    # print(
    #     f"OPENAI_API_KEY configured: {'Yes' if OPENAI_API_KEY else 'No'}",
    #     file=sys.stderr,
    # )

    # if not SUPABASE_JWT_SECRET:
    #     print(
    #         "WARNING: No SUPABASE_JWT_SECRET provided. API calls will likely fail.",
    #         file=sys.stderr,
    #     )
    #     print(
    #         "Please set SUPABASE_JWT_SECRET environment variable with a valid JWT token.",
    #         file=sys.stderr,
    #     )

    # if not OPENAI_API_KEY:
    #     print(
    #         "WARNING: No OPENAI_API_KEY provided. Multi-query generation will not work.",
    #         file=sys.stderr,
    #     )

    # Run stdio mode (default for Claude Desktop)
    mcp.run()


if __name__ == "__main__":
    main()
