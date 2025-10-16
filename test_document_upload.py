"""Test script to verify document upload and chunking works correctly."""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mcpserver.mcp_server import add_documents, list_collections, search_documents


async def test_document_upload():
    """Test document upload with different text sizes."""

    print("=" * 60)
    print("Document Upload Test")
    print("=" * 60)

    # Step 1: List collections
    print("\n1. Listing collections...")
    collections_result = await list_collections()
    print(collections_result)

    # Extract first collection ID from the result
    # Expected format: "Found N collections:\n- Name (ID)"
    lines = collections_result.split("\n")
    collection_id = None
    for line in lines:
        if " (" in line and ")" in line:
            # Extract ID from "- Name (collection-id)"
            collection_id = line.split("(")[1].split(")")[0].strip()
            print(f"\nUsing collection ID: {collection_id}")
            break

    if not collection_id:
        print("Error: No collections found. Please create a collection first.")
        return

    # Step 2: Test with small document
    print("\n2. Testing with small document (500 chars)...")
    small_text = "This is a test document. " * 20  # ~500 chars
    result = await add_documents(
        collection_id=collection_id,
        text=small_text,
        chunk_size=200,
        chunk_overlap=50,
        filename="small_test.txt"
    )
    print(result)

    # Step 3: Test with medium document
    print("\n3. Testing with medium document (5000 chars)...")
    medium_text = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
    ) * 50  # ~5000 chars
    result = await add_documents(
        collection_id=collection_id,
        text=medium_text,
        chunk_size=1000,
        chunk_overlap=200,
        filename="medium_test.txt"
    )
    print(result)

    # Step 4: Test with large document
    print("\n4. Testing with large document (50,000 chars)...")
    large_text = (
        "This is a comprehensive test document that contains multiple paragraphs. "
        "Each paragraph discusses different aspects of the system. "
        "The purpose is to test the chunking mechanism thoroughly. "
        "We want to ensure that all content is properly indexed. "
        "This will help us verify that the document processing pipeline works correctly. "
    ) * 500  # ~50,000 chars
    result = await add_documents(
        collection_id=collection_id,
        text=large_text,
        chunk_size=2000,
        chunk_overlap=400,
        filename="large_test.txt"
    )
    print(result)

    # Step 5: Search to verify content is indexed
    print("\n5. Searching for content...")
    search_result = await search_documents(
        collection_id=collection_id,
        query="test document",
        limit=5,
        search_type="semantic"
    )
    print(f"\nSearch results:\n{search_result}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check the API logs to see the detailed chunking information")
    print("2. Verify that chunk counts match text size / chunk_size")
    print("3. Try searching for specific phrases from your documents")


if __name__ == "__main__":
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_JWT_SECRET"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}")
        print("\nPlease set them in your .env file or environment.")
        sys.exit(1)

    asyncio.run(test_document_upload())
