"""Unit tests for document processor with chunk parameters."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import UploadFile
from langchain_core.documents import Document

from langconnect.services.document_processor import process_document


@pytest.mark.asyncio
async def test_process_document_with_default_chunk_params():
    """Test document processing with default chunk parameters."""
    # Create a mock UploadFile
    content = b"This is a test document. " * 50  # About 1250 characters
    file = MagicMock(spec=UploadFile)
    file.read = AsyncMock(return_value=content)
    file.filename = "test.txt"
    file.content_type = "text/plain"

    # Process document with default parameters
    documents = await process_document(file)

    assert isinstance(documents, list)
    assert len(documents) >= 2  # Should be split into at least 2 chunks

    # Check that each document is a LangChain Document
    for doc in documents:
        assert isinstance(doc, Document)
        assert hasattr(doc, "page_content")
        assert hasattr(doc, "metadata")
        assert "file_id" in doc.metadata

    # First chunk should be around 1000 characters (default chunk_size)
    assert 800 <= len(documents[0].page_content) <= 1200


@pytest.mark.asyncio
async def test_process_document_with_custom_chunk_size():
    """Test document processing with custom chunk_size."""
    content = b"ABCDEFGHIJ" * 30  # 300 characters
    file = MagicMock(spec=UploadFile)
    file.read = AsyncMock(return_value=content)
    file.filename = "test.txt"
    file.content_type = "text/plain"

    # Process with small chunk size
    documents = await process_document(file, chunk_size=50, chunk_overlap=0)

    assert len(documents) == 6  # 300 / 50 = 6 chunks

    # Each chunk should be exactly 50 characters
    for doc in documents:
        assert len(doc.page_content) == 50


@pytest.mark.asyncio
async def test_process_document_with_custom_overlap():
    """Test document processing with custom chunk_overlap."""
    content = b"0123456789" * 10  # 100 characters
    file = MagicMock(spec=UploadFile)
    file.read = AsyncMock(return_value=content)
    file.filename = "test.txt"
    file.content_type = "text/plain"

    # Process with overlap
    documents = await process_document(file, chunk_size=20, chunk_overlap=10)

    # With overlap, chunks should overlap by 10 characters
    assert len(documents) > 5  # More chunks due to overlap

    # Check that consecutive chunks overlap
    for i in range(len(documents) - 1):
        chunk1 = documents[i].page_content
        chunk2 = documents[i + 1].page_content
        # The last 10 chars of chunk1 should match the first 10 chars of chunk2
        assert chunk1[-10:] == chunk2[:10]


@pytest.mark.asyncio
async def test_process_document_with_metadata():
    """Test document processing with metadata."""
    content = b"Document with metadata"
    file = MagicMock(spec=UploadFile)
    file.read = AsyncMock(return_value=content)
    file.filename = "metadata_test.txt"
    file.content_type = "text/plain"

    metadata = {"author": "test_user", "category": "test"}

    documents = await process_document(file, metadata=metadata)

    assert len(documents) == 1  # Short content, single chunk
    doc = documents[0]

    # Check metadata is preserved
    assert doc.metadata["author"] == "test_user"
    assert doc.metadata["category"] == "test"
    assert "file_id" in doc.metadata


@pytest.mark.asyncio
async def test_process_document_large_chunk_size():
    """Test with chunk_size larger than content."""
    content = b"Short content"
    file = MagicMock(spec=UploadFile)
    file.read = AsyncMock(return_value=content)
    file.filename = "short.txt"
    file.content_type = "text/plain"

    # Process with very large chunk size
    documents = await process_document(file, chunk_size=5000)

    assert len(documents) == 1  # All content in one chunk
    assert documents[0].page_content == "Short content"


@pytest.mark.asyncio
async def test_process_document_markdown_file():
    """Test processing markdown files."""
    content = b"# Title\n\nThis is a **markdown** document.\n\n- Item 1\n- Item 2"
    file = MagicMock(spec=UploadFile)
    file.read = AsyncMock(return_value=content)
    file.filename = "test.md"
    file.content_type = "text/markdown"

    documents = await process_document(file, chunk_size=100, chunk_overlap=20)

    assert len(documents) >= 1
    # Content should be preserved as-is
    full_content = "".join(doc.page_content for doc in documents)
    assert "# Title" in full_content
    assert "**markdown**" in full_content


@pytest.mark.asyncio
async def test_process_document_with_file_id_generation():
    """Test that each processing generates a unique file_id."""
    content = b"Test content"

    # Process same content twice
    file1 = MagicMock(spec=UploadFile)
    file1.read = AsyncMock(return_value=content)
    file1.filename = "test.txt"
    file1.content_type = "text/plain"

    file2 = MagicMock(spec=UploadFile)
    file2.read = AsyncMock(return_value=content)
    file2.filename = "test.txt"
    file2.content_type = "text/plain"

    docs1 = await process_document(file1)
    docs2 = await process_document(file2)

    # Each processing should have a unique file_id
    file_id1 = docs1[0].metadata["file_id"]
    file_id2 = docs2[0].metadata["file_id"]

    assert file_id1 != file_id2
    assert len(file_id1) == 36  # UUID string length
    assert len(file_id2) == 36


@pytest.mark.asyncio
async def test_process_document_empty_file():
    """Test processing empty file."""
    content = b""
    file = MagicMock(spec=UploadFile)
    file.read = AsyncMock(return_value=content)
    file.filename = "empty.txt"
    file.content_type = "text/plain"

    documents = await process_document(file)

    # Empty file might result in no documents after splitting
    # This is expected behavior from the text splitter
    assert len(documents) == 0


@pytest.mark.asyncio
async def test_process_document_octet_stream_with_extension():
    """Test processing files with application/octet-stream mimetype."""
    content = b"Markdown content with unknown mimetype"
    file = MagicMock(spec=UploadFile)
    file.read = AsyncMock(return_value=content)
    file.filename = "test.md"
    file.content_type = "application/octet-stream"

    documents = await process_document(file)

    assert len(documents) >= 1
    assert documents[0].page_content == "Markdown content with unknown mimetype"
