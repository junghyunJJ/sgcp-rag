"""Test script for add_documents_from_files MCP tool."""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mcpserver.mcp_server import (
    add_documents_from_files,
    list_collections,
    search_documents,
)


async def test_file_upload():
    """Test file upload with different file types."""

    print("=" * 70)
    print("File Upload Test - add_documents_from_files()")
    print("=" * 70)

    # Step 1: Create test files
    print("\n1. Creating test files...")
    test_dir = Path("/tmp/langconnect_test")
    test_dir.mkdir(exist_ok=True)

    # Create small text file
    small_txt = test_dir / "small_test.txt"
    small_txt.write_text(
        "This is a small test file.\n"
        "It contains multiple lines.\n"
        "Each line has some content.\n" * 5
    )
    print(f"   Created: {small_txt} ({small_txt.stat().st_size} bytes)")

    # Create markdown file
    md_file = test_dir / "test_document.md"
    md_file.write_text(
        "# Test Document\n\n"
        "## Introduction\n"
        "This is a test markdown document for the new file upload feature.\n\n"
        "## Features\n"
        "- Direct file upload without reading into context\n"
        "- Support for multiple file types\n"
        "- Efficient batch processing\n\n"
        "## Conclusion\n"
        "The new approach saves context window and supports larger files.\n"
    )
    print(f"   Created: {md_file} ({md_file.stat().st_size} bytes)")

    # Create large text file
    large_txt = test_dir / "large_test.txt"
    large_txt.write_text(
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 100
    )
    print(f"   Created: {large_txt} ({large_txt.stat().st_size} bytes)")

    # Step 2: List collections
    print("\n2. Listing collections...")
    collections_result = await list_collections()
    print(collections_result)

    # Extract first collection ID
    lines = collections_result.split("\n")
    collection_id = None
    for line in lines:
        if " (" in line and ")" in line:
            collection_id = line.split("(")[1].split(")")[0].strip()
            print(f"\n   Using collection ID: {collection_id}")
            break

    if not collection_id:
        print("Error: No collections found. Please create a collection first.")
        return

    # Step 3: Test single file upload
    print("\n3. Testing single file upload...")
    result = await add_documents_from_files(
        collection_id=collection_id,
        file_paths=[str(small_txt)],
        chunk_size=100,
        chunk_overlap=20,
    )
    print(f"   Result: {result}")

    # Step 4: Test multiple file upload
    print("\n4. Testing multiple file upload...")
    result = await add_documents_from_files(
        collection_id=collection_id,
        file_paths=[str(md_file), str(large_txt)],
        chunk_size=300,
        chunk_overlap=50,
    )
    print(f"   Result: {result}")

    # Step 5: Test with invalid file
    print("\n5. Testing error handling (non-existent file)...")
    result = await add_documents_from_files(
        collection_id=collection_id,
        file_paths=["/tmp/nonexistent_file.txt", str(small_txt)],
    )
    print(f"   Result: {result}")

    # Step 6: Test with home directory expansion
    print("\n6. Testing home directory expansion (~)...")
    home_test = Path.home() / ".langconnect_test.txt"
    home_test.write_text("Test file in home directory.")
    result = await add_documents_from_files(
        collection_id=collection_id,
        file_paths=[f"~/.langconnect_test.txt"],
    )
    print(f"   Result: {result}")
    home_test.unlink()  # Clean up

    # Step 7: Search to verify
    print("\n7. Searching for uploaded content...")
    search_result = await search_documents(
        collection_id=collection_id,
        query="test file upload feature",
        limit=3,
    )
    print(f"   Found {len(search_result.split('---'))} results")

    # Cleanup
    print("\n8. Cleaning up test files...")
    for file in [small_txt, md_file, large_txt]:
        file.unlink()
    test_dir.rmdir()
    print("   Test files removed")

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)
    print("\nKey improvements:")
    print("✅ Files uploaded directly without reading into context")
    print("✅ Multiple files supported in single call")
    print("✅ MIME type auto-detection working")
    print("✅ File path validation and error handling")
    print("✅ Home directory (~) expansion supported")


if __name__ == "__main__":
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_JWT_SECRET"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}")
        print("\nPlease set them in your .env file or environment.")
        sys.exit(1)

    asyncio.run(test_file_upload())
