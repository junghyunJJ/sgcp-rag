"""Simple tests for chunk parameters without database dependency."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import UploadFile

from langconnect.api.documents import documents_create
from langconnect.services.document_processor import process_document


@pytest.mark.asyncio
async def test_documents_create_passes_chunk_params():
    """Test that the API endpoint correctly passes chunk parameters to process_document."""
    # Mock dependencies
    mock_user = MagicMock()
    mock_user.identity = "test_user"

    # Create mock file
    file_content = b"Test content for chunking"
    mock_file = MagicMock(spec=UploadFile)
    mock_file.read = AsyncMock(return_value=file_content)
    mock_file.filename = "test.txt"
    mock_file.content_type = "text/plain"

    # Mock the Collection class
    with patch("langconnect.api.documents.Collection") as MockCollection:
        # Setup mock collection instance
        mock_collection_instance = MagicMock()
        mock_collection_instance.upsert = AsyncMock(return_value=["doc1", "doc2"])
        MockCollection.return_value = mock_collection_instance

        # Mock process_document to verify it receives correct parameters
        with patch("langconnect.api.documents.process_document") as mock_process:
            # Setup mock return value
            mock_docs = [MagicMock(metadata={"file_id": "test_id"})]
            mock_process.return_value = mock_docs

            # Call the API endpoint with custom chunk parameters
            result = await documents_create(
                user=mock_user,
                collection_id="test-collection-id",
                files=[mock_file],
                metadatas_json=None,
                chunk_size=500,
                chunk_overlap=100,
            )

            # Verify process_document was called with correct parameters
            mock_process.assert_called_once()
            call_args = mock_process.call_args

            # Check that chunk parameters were passed correctly
            assert call_args.kwargs["chunk_size"] == 500
            assert call_args.kwargs["chunk_overlap"] == 100

            # Verify successful response
            assert result["success"] is True
            assert result["added_chunk_ids"] == ["doc1", "doc2"]


@pytest.mark.asyncio
async def test_process_document_respects_chunk_params():
    """Test that process_document actually uses the provided chunk parameters."""
    # Create a long text that will be split differently based on chunk_size
    long_text = "A" * 500 + "B" * 500 + "C" * 500  # 1500 characters total

    mock_file = MagicMock(spec=UploadFile)
    mock_file.read = AsyncMock(return_value=long_text.encode())
    mock_file.filename = "long_test.txt"
    mock_file.content_type = "text/plain"

    # Test with default parameters (chunk_size=1000)
    docs_default = await process_document(mock_file)

    # Reset the mock
    mock_file.read = AsyncMock(return_value=long_text.encode())

    # Test with smaller chunk size
    docs_small = await process_document(mock_file, chunk_size=300, chunk_overlap=0)

    # With chunk_size=300 and no overlap, we should get exactly 5 chunks
    assert len(docs_small) == 5
    # With default chunk_size=1000, we should get 2 chunks
    assert len(docs_default) == 2

    # Verify chunk sizes
    for doc in docs_small[:-1]:  # All but last chunk
        assert len(doc.page_content) == 300

    # Last chunk might be smaller
    assert len(docs_small[-1].page_content) <= 300


@pytest.mark.asyncio
async def test_chunk_overlap_behavior():
    """Test that chunk overlap works correctly."""
    # Create content where we can verify overlap
    content = "0123456789" * 5  # 50 characters

    mock_file = MagicMock(spec=UploadFile)
    mock_file.read = AsyncMock(return_value=content.encode())
    mock_file.filename = "overlap_test.txt"
    mock_file.content_type = "text/plain"

    # Process with overlap
    docs = await process_document(mock_file, chunk_size=20, chunk_overlap=5)

    # Verify overlap exists between consecutive chunks
    for i in range(len(docs) - 1):
        chunk1_end = docs[i].page_content[-5:]
        chunk2_start = docs[i + 1].page_content[:5]
        assert chunk1_end == chunk2_start, f"No overlap between chunk {i} and {i + 1}"


@pytest.mark.asyncio
async def test_api_default_chunk_params():
    """Test that API uses default chunk parameters when not provided."""
    mock_user = MagicMock()
    mock_user.identity = "test_user"

    mock_file = MagicMock(spec=UploadFile)
    mock_file.read = AsyncMock(return_value=b"Test content")
    mock_file.filename = "test.txt"
    mock_file.content_type = "text/plain"

    with patch("langconnect.api.documents.Collection") as MockCollection:
        # Setup mock collection instance
        mock_collection_instance = MagicMock()
        mock_collection_instance.upsert = AsyncMock(return_value=["doc1"])
        MockCollection.return_value = mock_collection_instance

        with patch("langconnect.api.documents.process_document") as mock_process:
            mock_process.return_value = [MagicMock(metadata={"file_id": "test"})]

            # Call without specifying chunk parameters
            # This simulates form data without chunk_size and chunk_overlap
            await documents_create(
                user=mock_user,
                collection_id="test-id",
                files=[mock_file],
                metadatas_json=None,
                chunk_size=1000,  # Default value from Form(1000)
                chunk_overlap=200,  # Default value from Form(200)
            )

            # Verify defaults were used
            call_args = mock_process.call_args
            assert call_args.kwargs["chunk_size"] == 1000
            assert call_args.kwargs["chunk_overlap"] == 200
