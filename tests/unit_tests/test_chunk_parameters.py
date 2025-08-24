"""Tests for custom chunk_size and chunk_overlap parameters in document processing."""

import json

from tests.unit_tests.fixtures import (
    get_async_test_client,
)

USER_1_HEADERS = {
    "Authorization": "Bearer user1",
}


async def test_documents_create_with_custom_chunk_size() -> None:
    """Test creating documents with custom chunk_size parameter."""
    async with get_async_test_client() as client:
        # Create a collection first
        collection_name = "chunk_size_test_col"
        col_payload = {"name": collection_name, "metadata": {"purpose": "chunk-test"}}
        create_col = await client.post(
            "/collections", json=col_payload, headers=USER_1_HEADERS
        )
        assert create_col.status_code == 201
        collection_data = create_col.json()
        collection_id = collection_data["uuid"]

        # Prepare a long text file that will be chunked
        long_text = "This is a test sentence. " * 100  # About 2500 characters
        files = [("files", ("long_test.txt", long_text.encode(), "text/plain"))]

        # Test with small chunk size (should create more chunks)
        small_chunk_size = 200
        resp = await client.post(
            f"/collections/{collection_id}/documents",
            files=files,
            data={"chunk_size": str(small_chunk_size), "chunk_overlap": "0"},
            headers=USER_1_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        small_chunk_ids = data["added_chunk_ids"]
        assert isinstance(small_chunk_ids, list)
        assert len(small_chunk_ids) > 10  # Should have many chunks with small size

        # List documents to verify chunk sizes
        list_resp = await client.get(
            f"/collections/{collection_id}/documents", headers=USER_1_HEADERS
        )
        assert list_resp.status_code == 200
        docs = list_resp.json()

        # Verify that chunks are approximately the specified size
        for doc in docs:
            content_length = len(doc["content"])
            assert content_length <= small_chunk_size + 50  # Allow some buffer


async def test_documents_create_with_custom_chunk_overlap() -> None:
    """Test creating documents with custom chunk_overlap parameter."""
    async with get_async_test_client() as client:
        # Create a collection
        collection_name = "chunk_overlap_test_col"
        col_payload = {"name": collection_name, "metadata": {"purpose": "overlap-test"}}
        create_col = await client.post(
            "/collections", json=col_payload, headers=USER_1_HEADERS
        )
        assert create_col.status_code == 201
        collection_data = create_col.json()
        collection_id = collection_data["uuid"]

        # Prepare a text file
        text_content = "ABCDEFGHIJ" * 50  # 500 characters of repeating pattern
        files = [("files", ("overlap_test.txt", text_content.encode(), "text/plain"))]

        # Test with custom chunk overlap
        chunk_size = 100
        chunk_overlap = 50
        resp = await client.post(
            f"/collections/{collection_id}/documents",
            files=files,
            data={"chunk_size": str(chunk_size), "chunk_overlap": str(chunk_overlap)},
            headers=USER_1_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        chunk_ids = data["added_chunk_ids"]
        assert isinstance(chunk_ids, list)

        # With overlap, we should have more chunks than without overlap
        expected_chunks_with_overlap = len(text_content) // (chunk_size - chunk_overlap)
        assert len(chunk_ids) >= expected_chunks_with_overlap - 1


async def test_documents_create_with_default_chunk_parameters() -> None:
    """Test that default chunk parameters work correctly."""
    async with get_async_test_client() as client:
        # Create a collection
        collection_name = "default_chunk_test_col"
        col_payload = {"name": collection_name, "metadata": {}}
        create_col = await client.post(
            "/collections", json=col_payload, headers=USER_1_HEADERS
        )
        assert create_col.status_code == 201
        collection_data = create_col.json()
        collection_id = collection_data["uuid"]

        # Prepare a text file
        text_content = "Test content. " * 100  # About 1400 characters
        files = [("files", ("default_test.txt", text_content.encode(), "text/plain"))]

        # Create document without specifying chunk parameters (should use defaults)
        resp = await client.post(
            f"/collections/{collection_id}/documents",
            files=files,
            headers=USER_1_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

        # List documents to check chunk size
        list_resp = await client.get(
            f"/collections/{collection_id}/documents", headers=USER_1_HEADERS
        )
        assert list_resp.status_code == 200
        docs = list_resp.json()

        # With default chunk_size=1000, we should have at least 2 chunks
        assert len(docs) >= 2
        # First chunk should be close to 1000 characters
        assert 800 <= len(docs[0]["content"]) <= 1200


async def test_documents_create_with_large_chunk_size() -> None:
    """Test creating documents with a very large chunk_size."""
    async with get_async_test_client() as client:
        # Create a collection
        collection_name = "large_chunk_test_col"
        col_payload = {"name": collection_name, "metadata": {}}
        create_col = await client.post(
            "/collections", json=col_payload, headers=USER_1_HEADERS
        )
        assert create_col.status_code == 201
        collection_data = create_col.json()
        collection_id = collection_data["uuid"]

        # Prepare a medium-sized text file
        text_content = "Large chunk test. " * 50  # About 900 characters
        files = [
            ("files", ("large_chunk_test.txt", text_content.encode(), "text/plain"))
        ]

        # Use a very large chunk size (larger than the content)
        large_chunk_size = 5000
        resp = await client.post(
            f"/collections/{collection_id}/documents",
            files=files,
            data={"chunk_size": str(large_chunk_size), "chunk_overlap": "0"},
            headers=USER_1_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        chunk_ids = data["added_chunk_ids"]

        # Should have only one chunk since content is smaller than chunk_size
        assert len(chunk_ids) == 1

        # Verify the entire content is in one chunk
        list_resp = await client.get(
            f"/collections/{collection_id}/documents", headers=USER_1_HEADERS
        )
        assert list_resp.status_code == 200
        docs = list_resp.json()
        assert len(docs) == 1
        assert docs[0]["content"] == text_content


async def test_documents_create_with_multiple_files_and_custom_chunks() -> None:
    """Test creating multiple documents with custom chunk parameters."""
    async with get_async_test_client() as client:
        # Create a collection
        collection_name = "multi_file_chunk_test"
        col_payload = {"name": collection_name, "metadata": {}}
        create_col = await client.post(
            "/collections", json=col_payload, headers=USER_1_HEADERS
        )
        assert create_col.status_code == 201
        collection_data = create_col.json()
        collection_id = collection_data["uuid"]

        # Prepare multiple files with different content lengths
        files = [
            ("files", ("short.txt", b"Short content", "text/plain")),
            ("files", ("medium.txt", ("Medium content. " * 50).encode(), "text/plain")),
            ("files", ("long.txt", ("Long content. " * 200).encode(), "text/plain")),
        ]

        # Prepare metadata for each file
        metadata = [
            {"source": "short.txt", "type": "short"},
            {"source": "medium.txt", "type": "medium"},
            {"source": "long.txt", "type": "long"},
        ]
        metadata_json = json.dumps(metadata)

        # Create documents with custom chunk parameters
        chunk_size = 300
        chunk_overlap = 50
        resp = await client.post(
            f"/collections/{collection_id}/documents",
            files=files,
            data={
                "metadatas_json": metadata_json,
                "chunk_size": str(chunk_size),
                "chunk_overlap": str(chunk_overlap),
            },
            headers=USER_1_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        chunk_ids = data["added_chunk_ids"]

        # Should have multiple chunks (at least one per file)
        assert len(chunk_ids) >= 3

        # Verify chunks retain metadata
        list_resp = await client.get(
            f"/collections/{collection_id}/documents", headers=USER_1_HEADERS
        )
        assert list_resp.status_code == 200
        docs = list_resp.json()

        # Check that metadata is preserved in chunks
        sources_found = set()
        for doc in docs:
            if "source" in doc["metadata"]:
                sources_found.add(doc["metadata"]["source"])

        # All three sources should be present
        assert "short.txt" in sources_found
        assert "medium.txt" in sources_found
        assert "long.txt" in sources_found


async def test_documents_create_with_zero_overlap() -> None:
    """Test creating documents with zero chunk overlap."""
    async with get_async_test_client() as client:
        # Create a collection
        collection_name = "zero_overlap_test"
        col_payload = {"name": collection_name, "metadata": {}}
        create_col = await client.post(
            "/collections", json=col_payload, headers=USER_1_HEADERS
        )
        assert create_col.status_code == 201
        collection_data = create_col.json()
        collection_id = collection_data["uuid"]

        # Prepare a text file with exact content to test chunking
        # Create content that's exactly 300 characters
        text_content = "A" * 100 + "B" * 100 + "C" * 100
        files = [("files", ("zero_overlap.txt", text_content.encode(), "text/plain"))]

        # Create document with chunk_size=100 and overlap=0
        chunk_size = 100
        resp = await client.post(
            f"/collections/{collection_id}/documents",
            files=files,
            data={"chunk_size": str(chunk_size), "chunk_overlap": "0"},
            headers=USER_1_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        chunk_ids = data["added_chunk_ids"]

        # Should have exactly 3 chunks with no overlap
        assert len(chunk_ids) == 3

        # Verify chunks don't overlap
        list_resp = await client.get(
            f"/collections/{collection_id}/documents", headers=USER_1_HEADERS
        )
        assert list_resp.status_code == 200
        docs = list_resp.json()

        # Sort by content to ensure order
        docs_sorted = sorted(docs, key=lambda x: x["content"])

        # Each chunk should contain only one type of character
        assert "A" in docs_sorted[0]["content"] and "B" not in docs_sorted[0]["content"]
        assert "B" in docs_sorted[1]["content"]
        assert "C" in docs_sorted[2]["content"] and "B" not in docs_sorted[2]["content"]


async def test_documents_create_with_edge_case_parameters() -> None:
    """Test creating documents with edge case chunk parameters."""
    async with get_async_test_client() as client:
        # Create a collection
        collection_name = "edge_case_chunk_test"
        col_payload = {"name": collection_name, "metadata": {}}
        create_col = await client.post(
            "/collections", json=col_payload, headers=USER_1_HEADERS
        )
        assert create_col.status_code == 201
        collection_data = create_col.json()
        collection_id = collection_data["uuid"]

        # Test 1: chunk_overlap equal to chunk_size (should still work)
        text_content = "Test content for edge cases. " * 20
        files = [("files", ("edge_test.txt", text_content.encode(), "text/plain"))]

        resp = await client.post(
            f"/collections/{collection_id}/documents",
            files=files,
            data={"chunk_size": "100", "chunk_overlap": "100"},
            headers=USER_1_HEADERS,
        )
        assert resp.status_code == 200

        # Test 2: Very small chunk_size
        resp2 = await client.post(
            f"/collections/{collection_id}/documents",
            files=files,
            data={"chunk_size": "10", "chunk_overlap": "5"},
            headers=USER_1_HEADERS,
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["success"] is True
        # Should create many small chunks
        assert len(data2["added_chunk_ids"]) > 20
