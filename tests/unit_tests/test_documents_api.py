import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from uuid import UUID

import pytest
from fastapi import HTTPException
from langchain_core.documents import Document
from pydantic import ValidationError

from langconnect.database.collections import Collection
from langconnect.models.document import SearchQuery
from tests.unit_tests.fixtures import (
    get_async_test_client,
)

pytestmark = pytest.mark.asyncio


class _FakeVectorStore:
    def __init__(self, results: list[tuple[Document, float]]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    def similarity_search_with_score(
        self,
        query: str,
        k: int,
        filter: dict[str, object] | None = None,  # noqa: A002
    ) -> list[tuple[Document, float]]:
        self.calls.append({"query": query, "k": k, "filter": filter})
        results = self.results
        if filter:
            results = [
                (doc, distance)
                for doc, distance in results
                if all(doc.metadata.get(key) == value for key, value in filter.items())
            ]
        return results[:k]


class _FakeDbConnection:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.calls: list[tuple[object, ...]] = []

    async def fetch(self, *_args: object) -> list[dict[str, object]]:
        self.calls.append(_args)
        limit = int(_args[-1])
        return self.rows[:limit]


def _doc(doc_id: str, content: str, **metadata: object) -> Document:
    return Document(id=doc_id, page_content=content, metadata=metadata)


def _keyword_row(
    doc_id: object,
    content: str,
    score: float,
    **metadata: object,
) -> dict[str, object]:
    return {
        "id": doc_id,
        "page_content": content,
        "metadata": json.dumps(metadata),
        "score": score,
    }


def _patch_collection_details(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_details(self: Collection) -> dict[str, object]:
        return {"uuid": self.collection_id, "table_id": "test_table"}

    monkeypatch.setattr(Collection, "_get_details_or_raise", fake_details)


def _patch_vectorstore(
    monkeypatch: pytest.MonkeyPatch,
    store: _FakeVectorStore,
) -> None:
    monkeypatch.setattr(
        "langconnect.database.collections.get_vectorstore",
        lambda *args, **kwargs: store,
    )


def _patch_keyword_rows(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, object]],
) -> _FakeDbConnection:
    connection = _FakeDbConnection(rows)

    @asynccontextmanager
    async def fake_connection() -> AsyncGenerator[_FakeDbConnection, None]:
        yield connection

    monkeypatch.setattr(
        "langconnect.database.collections.get_db_connection",
        fake_connection,
    )
    return connection


async def test_documents_create_and_list_and_delete_and_search() -> None:
    """Test creating, listing, deleting, and searching documents."""
    async with get_async_test_client() as client:
        # Create a collection for documents
        collection_name = "docs_test_col"
        col_payload = {"name": collection_name, "metadata": {"purpose": "doc-test"}}
        create_col = await client.post(
            "/collections", json=col_payload
        )
        assert create_col.status_code == 201
        collection_data = create_col.json()
        collection_id = collection_data["uuid"]

        # Prepare a simple text file
        file_content = b"Hello world. This is a test document."
        files = [("files", ("test.txt", file_content, "text/plain"))]
        # Create documents without metadata
        resp = await client.post(
            f"/collections/{collection_id}/documents",
            files=files,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        # added_chunk_ids should be a non-empty list of UUIDs
        ids = data["added_chunk_ids"]
        assert isinstance(ids, list)
        assert ids
        for chunk_id in ids:
            # Validate each is a UUID string
            UUID(chunk_id)

        # List documents in collection, default limit 10
        list_resp = await client.get(
            f"/collections/{collection_id}/documents"
        )
        assert list_resp.status_code == 200
        docs = list_resp.json()
        assert isinstance(docs, list)
        assert docs
        # Each doc should have id and text fields

        assert len(docs) == 1
        assert docs[0]["content"] == "Hello world. This is a test document."

        # Search documents with a valid query
        search_payload = {"query": "test document", "limit": 5, "min_score": 0}
        search_resp = await client.post(
            f"/collections/{collection_id}/documents/search",
            json=search_payload,
        )
        assert search_resp.status_code == 200
        results = search_resp.json()
        assert isinstance(results, list)
        # Each result should have id, score, text
        assert len(results) == 1
        assert results[0] == {
            "id": docs[0]["id"],
            "score": results[0]["score"],
            "page_content": "Hello world. This is a test document.",
            "metadata": {
                "file_id": docs[0]["metadata"]["file_id"],
                "source": None,
            },
        }

        # Delete a document
        doc_id = docs[0]["id"]
        del_resp = await client.delete(
            f"/collections/{collection_id}/documents/{doc_id}",
        )
        assert del_resp.status_code == 200
        assert del_resp.json() == {"success": True}

        # Delete non-existent document gracefully
        del_resp2 = await client.delete(
            f"/collections/{collection_id}/documents/{doc_id}",
        )
        # Should still return success True or 200/204; here assume 200
        assert del_resp2.status_code in (200, 204)


async def test_semantic_search_rejects_default_low_score_no_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default semantic search should not return unrelated nearest neighbors."""
    _patch_collection_details(monkeypatch)
    store = _FakeVectorStore([
        (
            _doc(
                "irrelevant-1",
                "Unrelated biology pathway text without agent content.",
                source="distractor.pdf",
            ),
            0.54,
        ),
        (
            _doc(
                "irrelevant-2",
                "Another unrelated chunk about metabolomics.",
                source="distractor.pdf",
            ),
            0.60,
        ),
    ])
    _patch_vectorstore(monkeypatch, store)

    results = await Collection("collection-id").search(
        "zzzz qwerty asdfgh unrelated nonsense",
        limit=10,
        search_type="semantic",
    )

    assert results == []


async def test_semantic_search_rejects_observed_nonsense_score_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default semantic threshold should reject observed nonsense scores near 0.67."""
    _patch_collection_details(monkeypatch)
    store = _FakeVectorStore([
        (
            _doc(
                "nonsense-boundary",
                "Nearest-neighbor text unrelated to the query.",
                source="distractor.pdf",
            ),
            0.4925,  # similarity ~= 0.67
        ),
    ])
    _patch_vectorstore(monkeypatch, store)

    results = await Collection("collection-id").search(
        "zzzz qwerty asdfgh unrelated nonsense",
        limit=10,
        search_type="semantic",
    )

    assert results == []


async def test_semantic_search_allows_explicit_lower_min_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callers can opt into lower semantic thresholds without changing limit."""
    _patch_collection_details(monkeypatch)
    store = _FakeVectorStore([
        (
            _doc(
                "low-score-match",
                "A weak but caller-accepted semantic match.",
                source="weak.pdf",
            ),
            0.54,
        ),
    ])
    _patch_vectorstore(monkeypatch, store)

    results = await Collection("collection-id").search(
        "weak match",
        limit=10,
        search_type="semantic",
        min_score=0.5,
    )

    assert [result["id"] for result in results] == ["low-score-match"]


async def test_semantic_search_allows_explicit_boundary_min_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callers can still opt into accepting a score near the default boundary."""
    _patch_collection_details(monkeypatch)
    store = _FakeVectorStore([
        (
            _doc(
                "boundary-match",
                "A caller-accepted lower confidence semantic match.",
                source="weak.pdf",
            ),
            0.4925,  # similarity ~= 0.67
        ),
    ])
    _patch_vectorstore(monkeypatch, store)

    results = await Collection("collection-id").search(
        "weak match",
        limit=10,
        search_type="semantic",
        min_score=0.66,
    )

    assert [result["id"] for result in results] == ["boundary-match"]


async def test_semantic_metadata_filter_applies_before_candidate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filtered semantic results should not depend on unfiltered neighbors."""
    _patch_collection_details(monkeypatch)
    distractors = [
        (
            _doc(
                f"distractor-{index}",
                f"Agent skill distractor from another source {index}",
                source="other.pdf",
            ),
            0.02,
        )
        for index in range(30)
    ]
    skillfoundry_matches = [
        (
            _doc(
                f"skillfoundry-{index}",
                f"Agent skill memory repository chunk {index}",
                source="skillfoundry.pdf",
            ),
            0.05,
        )
        for index in range(30)
    ]
    store = _FakeVectorStore([*distractors, *skillfoundry_matches])
    _patch_vectorstore(monkeypatch, store)

    limit_10 = await Collection("collection-id").search(
        "agent skill",
        limit=10,
        search_type="semantic",
        filter={"source": "skillfoundry.pdf"},
    )
    limit_50 = await Collection("collection-id").search(
        "agent skill",
        limit=50,
        search_type="semantic",
        filter={"source": "skillfoundry.pdf"},
    )

    assert [result["id"] for result in limit_10] == [
        result["id"] for result in limit_50[:10]
    ]
    assert len(limit_10) == 10
    assert all(
        result["metadata"]["source"] == "skillfoundry.pdf" for result in limit_10
    )
    assert store.calls[0]["filter"] == {"source": "skillfoundry.pdf"}


async def test_hybrid_search_rejects_default_low_score_no_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default hybrid search should not revive weak semantic candidates."""
    _patch_collection_details(monkeypatch)
    store = _FakeVectorStore([
        (
            _doc(
                "irrelevant-semantic",
                "Unrelated nearest-neighbor content.",
                source="distractor.pdf",
            ),
            0.54,
        ),
    ])
    _patch_vectorstore(monkeypatch, store)
    _patch_keyword_rows(monkeypatch, [])

    results = await Collection("collection-id").search(
        "zzzz qwerty asdfgh unrelated nonsense",
        limit=10,
        search_type="hybrid",
    )

    assert results == []


async def test_hybrid_search_preserves_strong_semantic_match_over_keyword_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid ranking should not demote strong semantic matches via normalization."""
    _patch_collection_details(monkeypatch)
    strong_semantic = _doc(
        "semantic-strong",
        "Agent skills persist reusable behavior across coding sessions.",
        source="skillfoundry.pdf",
    )
    keyword_only = _doc(
        "keyword-only",
        "The word agent appears many times without explaining skills.",
        source="keyword.pdf",
    )
    store = _FakeVectorStore([(strong_semantic, 0.05)])
    _patch_vectorstore(monkeypatch, store)
    _patch_keyword_rows(monkeypatch, [
        _keyword_row(
            "keyword-only",
            keyword_only.page_content,
            10.0,
            source="keyword.pdf",
        ),
        _keyword_row(
            "semantic-strong",
            strong_semantic.page_content,
            1.0,
            source="skillfoundry.pdf",
        ),
    ])

    results = await Collection("collection-id").search(
        "agent skill",
        limit=2,
        search_type="hybrid",
    )

    assert [result["id"] for result in results] == [
        "semantic-strong",
        "keyword-only",
    ]
    assert results[1]["score"] == pytest.approx(0.3)
    assert results[1]["metadata"] == {"source": "keyword.pdf"}


async def test_hybrid_search_dedupes_mixed_id_types_and_combines_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Semantic and keyword hits for the same id should return one fused row."""
    _patch_collection_details(monkeypatch)
    semantic_doc = _doc(
        "42",
        "Semantic content for a shared document id.",
        source="semantic.pdf",
    )
    store = _FakeVectorStore([(semantic_doc, 0.25)])
    _patch_vectorstore(monkeypatch, store)
    _patch_keyword_rows(monkeypatch, [
        _keyword_row(
            42,
            "Keyword content for the same shared document id.",
            2.0,
            source="keyword.pdf",
        ),
        _keyword_row(
            "keyword-only",
            "Keyword-only content should remain in hybrid results.",
            1.0,
            source="keyword-only.pdf",
        ),
    ])

    results = await Collection("collection-id").search(
        "shared document",
        limit=10,
        search_type="hybrid",
    )

    assert [result["id"] for result in results] == ["42", "keyword-only"]
    assert results[0]["page_content"] == semantic_doc.page_content
    assert results[0]["metadata"] == {"source": "semantic.pdf"}
    assert results[0]["score"] == pytest.approx(0.86)
    assert results[1]["score"] == pytest.approx(0.15)


@pytest.mark.parametrize("search_type", ["semantic", "keyword", "hybrid"])
async def test_search_rejects_unsupported_metadata_filter_shape(
    monkeypatch: pytest.MonkeyPatch,
    search_type: str,
) -> None:
    """Unsupported filter operators should fail with a clear HTTP 400."""
    _patch_collection_details(monkeypatch)
    _patch_vectorstore(monkeypatch, _FakeVectorStore([]))
    _patch_keyword_rows(monkeypatch, [])

    with pytest.raises(HTTPException) as exc_info:
        await Collection("collection-id").search(
            "agent skill",
            search_type=search_type,
            filter={"source": {"$ne": "skillfoundry.pdf"}},
        )

    assert exc_info.value.status_code == 400
    assert "Unsupported metadata filter" in str(exc_info.value.detail)


@pytest.mark.parametrize("search_type", ["semantic", "keyword", "hybrid"])
async def test_search_rejects_metadata_filter_keys_pgvector_cannot_apply(
    monkeypatch: pytest.MonkeyPatch,
    search_type: str,
) -> None:
    """Accepted metadata filter keys should be valid for every search backend."""
    _patch_collection_details(monkeypatch)
    _patch_vectorstore(monkeypatch, _FakeVectorStore([]))
    _patch_keyword_rows(monkeypatch, [])

    with pytest.raises(HTTPException) as exc_info:
        await Collection("collection-id").search(
            "agent skill",
            search_type=search_type,
            filter={"source-url": "skillfoundry.pdf"},
        )

    assert exc_info.value.status_code == 400
    assert "filter field names" in str(exc_info.value.detail)


@pytest.mark.parametrize("limit", [0, -1, 101])
async def test_collection_search_rejects_invalid_limits(limit: int) -> None:
    """Collection.search should guard direct callers from invalid limits."""
    with pytest.raises(HTTPException) as exc_info:
        await Collection("collection-id").search("agent skill", limit=limit)

    assert exc_info.value.status_code == 400
    assert "limit must be between" in str(exc_info.value.detail)


@pytest.mark.parametrize("limit", [0, -1, 101])
async def test_search_query_rejects_invalid_limits(limit: int) -> None:
    """REST search requests should validate limit before reaching retrieval."""
    with pytest.raises(ValidationError):
        SearchQuery(query="agent skill", limit=limit)


async def test_documents_create_with_invalid_metadata_json() -> None:
    """Test creating documents with invalid metadata JSON."""
    async with get_async_test_client() as client:
        # Create a collection
        col_name = "meta_test_col"
        collection_response = await client.post(
            "/collections",
            json={"name": col_name, "metadata": {}},
        )
        assert collection_response.status_code == 201
        collection_data = collection_response.json()
        collection_id = collection_data["uuid"]

        # Prepare file
        file_content = b"Sample"
        files = [("files", ("a.txt", file_content, "text/plain"))]
        # Provide invalid JSON
        resp = await client.post(
            f"/collections/{collection_id}/documents",
            files=files,
            data={"metadatas_json": "not-a-json"},
        )
        assert resp.status_code == 400


async def test_documents_search_empty_query() -> None:
    """Test searching documents with an empty query."""
    async with get_async_test_client() as client:
        # Create a collection for search test
        col_name = "search_test_col"
        collection_response = await client.post(
            "/collections",
            json={"name": col_name, "metadata": {}},
        )
        assert collection_response.status_code == 201
        collection_data = collection_response.json()
        collection_id = collection_data["uuid"]

        # Attempt search with empty query
        resp = await client.post(
            f"/collections/{collection_id}/documents/search",
            json={"query": "", "limit": 3},
        )
        assert resp.status_code == 400
        assert "Search query cannot be empty" in resp.json()["detail"]


async def test_documents_in_nonexistent_collection() -> None:
    """Test operations on documents in a non-existent collection."""
    async with get_async_test_client() as client:
        # Try listing documents in missing collection
        no_such_collection = "12345678-1234-5678-1234-567812345678"
        response = await client.get(
            f"/collections/{no_such_collection}/documents"
        )
        assert response.status_code == 404

        # Try uploading to a non existent collection
        file_content = b"X"
        files = [("files", ("x.txt", file_content, "text/plain"))]
        upload_resp = await client.post(
            f"/collections/{no_such_collection}/documents",
            files=files,
        )
        assert upload_resp.status_code == 404
        assert "Collection not found" in upload_resp.json()["detail"]

        # Try deleting from missing collection/document
        del_resp = await client.delete(
            f"/collections/{no_such_collection}/documents/abcdef",
        )
        assert del_resp.status_code == 404

        # Try search in missing collection
        search_resp = await client.post(
            f"/collections/{no_such_collection}/documents/search",
            json={"query": "foo"},
        )
        # Not found or 404
        assert search_resp.status_code == 404


async def test_documents_create_with_valid_text_file_and_metadata() -> None:
    """Test creating documents with a valid text file and metadata."""
    async with get_async_test_client() as client:
        # Create a collection first
        collection_name = "doc_test_with_metadata"
        collection_response = await client.post(
            "/collections",
            json={"name": collection_name, "metadata": {}},
        )
        assert collection_response.status_code == 201
        collection_data = collection_response.json()
        collection_id = collection_data["uuid"]

        # Prepare a text file with content
        file_content = b"This is a test document with metadata."
        files = [("files", ("metadata_test.txt", file_content, "text/plain"))]

        # Prepare metadata as JSON
        metadata = [{"source": "test", "author": "user1", "importance": "high"}]
        metadata_json = json.dumps(metadata)

        # Create document with metadata
        response = await client.post(
            f"/collections/{collection_id}/documents",
            files=files,
            data={"metadatas_json": metadata_json},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "added_chunk_ids" in data
        ids = data["added_chunk_ids"]
        assert isinstance(ids, list)
        assert len(ids) > 0

        # Verify each ID is a valid UUID
        for chunk_id in ids:
            UUID(chunk_id)  # This will raise an exception if invalid

        # Verify document was added by listing documents
        list_response = await client.get(
            f"/collections/{collection_id}/documents",
        )
        assert list_response.status_code == 200
        documents = list_response.json()
        assert len(documents) == 1

        # Verify metadata was attached
        doc = documents[0]
        assert "metadata" in doc
        assert "file_id" in doc["metadata"]
        # The file_id will be a new UUID, so we can't check the exact value


async def test_documents_create_with_valid_text_file_without_metadata() -> None:
    """Test creating documents with a valid text file without metadata."""
    async with get_async_test_client() as client:
        # Create a collection first
        collection_name = "doc_test_without_metadata"
        collection_response = await client.post(
            "/collections",
            json={"name": collection_name, "metadata": {}},
        )
        assert collection_response.status_code == 201
        collection_data = collection_response.json()
        collection_id = collection_data["uuid"]

        # Prepare a text file with content
        file_content = b"This is a test document without metadata."
        files = [("files", ("no_metadata_test.txt", file_content, "text/plain"))]

        # Create document without metadata
        response = await client.post(
            f"/collections/{collection_id}/documents",
            files=files,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "added_chunk_ids" in data
        ids = data["added_chunk_ids"]
        assert isinstance(ids, list)
        assert len(ids) > 0

        # Verify document was added by listing documents
        list_response = await client.get(
            f"/collections/{collection_id}/documents",
        )
        assert list_response.status_code == 200
        documents = list_response.json()
        assert len(documents) > 0
        # Verify content is in the document
        assert documents[0]["content"] == "This is a test document without metadata."


async def test_documents_create_with_empty_file() -> None:
    """Test creating documents with an empty file."""
    async with get_async_test_client() as client:
        # Create a collection first
        collection_name = "doc_test_empty_file"
        collection_response = await client.post(
            "/collections",
            json={"name": collection_name, "metadata": {}},
        )
        assert collection_response.status_code == 201
        collection_data = collection_response.json()
        collection_id = collection_data["uuid"]

        # Prepare an empty file
        file_content = b""
        files = [("files", ("empty.txt", file_content, "text/plain"))]

        # Create document with empty file
        response = await client.post(
            f"/collections/{collection_id}/documents",
            files=files,
        )

        # Empty files should be rejected with 400 Bad Request
        assert response.status_code == 400
        data = response.json()
        assert "Failed to process any documents" in data["detail"]


async def test_documents_create_with_invalid_metadata_format() -> None:
    """Test creating documents with invalid metadata format."""
    async with get_async_test_client() as client:
        # Create a collection first
        collection_name = "doc_test_invalid_metadata"
        collection_response = await client.post(
            "/collections",
            json={"name": collection_name, "metadata": {}},
        )
        assert collection_response.status_code == 201
        collection_data = collection_response.json()
        collection_id = collection_data["uuid"]

        # Prepare a text file with content
        file_content = b"This is a test document with invalid metadata."
        files = [("files", ("invalid_metadata.txt", file_content, "text/plain"))]

        # Invalid JSON format for metadata
        invalid_metadata = "not a json"

        # Create document with invalid metadata
        response = await client.post(
            f"/collections/{collection_id}/documents",
            files=files,
            data={"metadatas_json": invalid_metadata},
        )

        assert response.status_code == 400

        # Test with metadata that's not a list
        invalid_metadata_not_list = json.dumps({"key": "value"})
        response = await client.post(
            f"/collections/{collection_id}/documents",
            files=files,
            data={"metadatas_json": invalid_metadata_not_list},
        )

        assert response.status_code == 400


async def test_documents_create_with_non_existent_collection() -> None:
    """Test creating documents in a non-existent collection."""
    async with get_async_test_client() as client:
        # Prepare a text file with content
        file_content = b"This is a test document for a non-existent collection."
        files = [("files", ("nonexistent.txt", file_content, "text/plain"))]

        # Try to create document in a non-existent collection
        uuid = "12345678-1234-5678-1234-567812345678"
        response = await client.post(
            f"/collections/{uuid}/documents",
            files=files,
        )

        assert response.status_code == 404
        data = response.json()
        assert "Collection not found" in data["detail"]


async def test_documents_create_with_multiple_files():
    """Test creating documents with multiple files."""
    async with get_async_test_client() as client:
        # Create a collection first
        collection_name = "doc_test_multiple_files"
        collection_response = await client.post(
            "/collections",
            json={"name": collection_name, "metadata": {}},
        )
        assert collection_response.status_code == 201
        collection_data = collection_response.json()
        collection_id = collection_data["uuid"]

        # Prepare multiple files
        files = [
            ("files", ("file1.txt", b"Content of file 1", "text/plain")),
            ("files", ("file2.txt", b"Content of file 2", "text/plain")),
        ]

        # Create document with multiple files
        response = await client.post(
            f"/collections/{collection_id}/documents",
            files=files,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "added_chunk_ids" in data
        ids = data["added_chunk_ids"]
        assert isinstance(ids, list)
        # We should have at least 2 chunks (one for each file)
        assert len(ids) >= 2

        # Verify documents were added by listing documents
        list_response = await client.get(
            f"/collections/{collection_id}/documents",
        )
        assert list_response.status_code == 200
        documents = list_response.json()
        # The number of documents returned might not match the number of files
        # exactly, as documents are chunked and only one chunk per file_id is returned
        assert len(documents) > 0


async def test_documents_create_with_mismatched_metadata():
    """Test creating documents with metadata count not matching files count."""
    async with get_async_test_client() as client:
        # Create a collection first
        collection_name = "doc_test_mismatched_metadata"
        collection_response = await client.post(
            "/collections",
            json={"name": collection_name, "metadata": {}},
        )
        assert collection_response.status_code == 201
        collection_data = collection_response.json()
        collection_id = collection_data["uuid"]

        # Prepare multiple files
        files = [
            ("files", ("file1.txt", b"Content of file 1", "text/plain")),
            ("files", ("file2.txt", b"Content of file 2", "text/plain")),
        ]

        # Metadata with only one entry for two files
        metadata = [{"source": "test"}]
        metadata_json = json.dumps(metadata)

        # Create document with mismatched metadata
        response = await client.post(
            f"/collections/{collection_id}/documents",
            files=files,
            data={"metadatas_json": metadata_json},
        )

        assert response.status_code == 400
        data = response.json()
        assert "does not match number of files" in data["detail"]


async def test_documents_bulk_delete() -> None:
    """Test bulk deleting documents by document_ids and file_ids."""
    async with get_async_test_client() as client:
        # Create a collection
        collection_name = "bulk_delete_test_col"
        col_payload = {"name": collection_name}
        create_col = await client.post("/collections", json=col_payload)
        assert create_col.status_code == 201
        collection_id = create_col.json()["uuid"]

        # Upload two different files, resulting in multiple chunks
        files1 = [("files", ("file1.txt", b"first file content", "text/plain"))]
        meta1 = json.dumps([{"source": "file1.txt"}])
        await client.post(
            f"/collections/{collection_id}/documents",
            files=files1,
            data={"metadatas_json": meta1},
        )

        files2 = [("files", ("file2.txt", b"second file content", "text/plain"))]
        meta2 = json.dumps([{"source": "file2.txt"}])
        await client.post(
            f"/collections/{collection_id}/documents",
            files=files2,
            data={"metadatas_json": meta2},
        )

        # List all documents to get their IDs
        list_resp = await client.get(f"/collections/{collection_id}/documents?limit=100")
        assert list_resp.status_code == 200
        docs = list_resp.json()
        assert len(docs) == 2

        doc_ids = [doc["id"] for doc in docs]
        file_ids = [doc["metadata"]["file_id"] for doc in docs]

        # Bulk delete by document_ids
        del_payload_docs = {"document_ids": [doc_ids[0]]}
        del_resp_docs = await client.request(
            "DELETE",
            f"/collections/{collection_id}/documents",
            json=del_payload_docs,
        )
        assert del_resp_docs.status_code == 200
        assert del_resp_docs.json()["success"] is True
        assert del_resp_docs.json()["deleted_count"] == 1

        # Verify one document is left
        list_resp_after_doc_delete = await client.get(f"/collections/{collection_id}/documents")
        assert len(list_resp_after_doc_delete.json()) == 1

        # Bulk delete by file_ids
        del_payload_files = {"file_ids": [file_ids[1]]}
        del_resp_files = await client.request(
            "DELETE",
            f"/collections/{collection_id}/documents",
            json=del_payload_files,
        )
        assert del_resp_files.status_code == 200
        assert del_resp_files.json()["success"] is True
        assert del_resp_files.json()["deleted_count"] == 1

        # Verify no documents are left
        list_resp_after_file_delete = await client.get(f"/collections/{collection_id}/documents")
        assert list_resp_after_file_delete.json() == []
