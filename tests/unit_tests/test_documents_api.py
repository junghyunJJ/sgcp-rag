import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from io import BytesIO
from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi import HTTPException, UploadFile
from httpx import ASGITransport, AsyncClient
from langchain_core.documents import Document
from pydantic import ValidationError

import langconnect.api.documents as documents_api
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
    def __init__(
        self,
        rows: list[dict[str, object]],
        execute_results: list[str] | None = None,
    ) -> None:
        self.rows = rows
        self.execute_results = execute_results or []
        self.calls: list[tuple[object, ...]] = []

    async def fetch(self, *_args: object) -> list[dict[str, object]]:
        self.calls.append(_args)
        limit = int(_args[-1])
        return self.rows[:limit]

    async def execute(self, *_args: object) -> str:
        self.calls.append(_args)
        if not self.execute_results:
            return "DELETE 0"
        return self.execute_results.pop(0)


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


@pytest.fixture(autouse=True)
def _stub_llm_wiki_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid real LLM rebuilds in existing document API tests."""

    async def fake_rebuild(collection_id: str, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            model_dump=lambda mode="json": {
                "collection_id": collection_id,
                "status": "rebuilt",
                "source_page_count": 0,
                "concept_page_count": 0,
                "page_count": 0,
                "chunk_count": 0,
                "pack_path": f"llm_wiki/collections/{collection_id}.json",
            }
        )

    monkeypatch.setattr(
        documents_api,
        "rebuild_llm_wiki",
        fake_rebuild,
        raising=False,
    )


def _upload_file(name: str = "doc.txt", content: bytes = b"hello") -> UploadFile:
    return UploadFile(filename=name, file=BytesIO(content))


def _patch_collection_execute(
    monkeypatch: pytest.MonkeyPatch,
    execute_results: list[str],
) -> _FakeDbConnection:
    connection = _FakeDbConnection([], execute_results=execute_results)

    @asynccontextmanager
    async def fake_connection() -> AsyncGenerator[_FakeDbConnection, None]:
        yield connection

    monkeypatch.setattr(
        "langconnect.database.collections.get_db_connection",
        fake_connection,
    )
    return connection


async def test_collection_delete_returns_deleted_count_for_document_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collection.delete should return a count for document-id deletes."""
    _patch_collection_execute(monkeypatch, ["DELETE 1"])

    result = await Collection("collection-1").delete(document_id="doc-a")

    assert type(result) is int
    assert result == 1


async def test_collection_delete_returns_deleted_count_for_file_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collection.delete should return a count for file-id deletes."""
    _patch_collection_execute(monkeypatch, ["DELETE 3"])

    result = await Collection("collection-1").delete(file_id="file-a")

    assert type(result) is int
    assert result == 3


async def test_documents_delete_rebuilds_after_document_id_delete_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Document-id delete should await a collection-scoped wiki rebuild."""
    collection_id = UUID("00000000-0000-0000-0000-000000000001")
    events: list[str] = []
    rebuild_calls: list[str] = []

    class FakeCollection:
        def __init__(self, collection_id: str) -> None:
            self.collection_id = collection_id

        async def delete(self, **kwargs: object) -> int:
            assert kwargs == {"document_id": "doc-a"}
            events.append("deleted")
            return 1

    async def fake_rebuild(collection_id: str, **kwargs: object) -> SimpleNamespace:
        rebuild_calls.append(collection_id)
        events.append("rebuilt")
        return SimpleNamespace(model_dump=lambda mode="json": {})

    monkeypatch.setattr(documents_api, "Collection", FakeCollection)
    monkeypatch.setattr(documents_api, "rebuild_llm_wiki", fake_rebuild)

    response = await documents_api.documents_delete(
        collection_id=collection_id,
        document_id="doc-a",
        delete_by="document_id",
    )

    assert response == {"success": True}
    assert set(response.keys()) == {"success"}
    assert rebuild_calls == [str(collection_id)]
    assert events == ["deleted", "rebuilt"]


async def test_documents_delete_rebuilds_after_file_id_delete_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """File-id delete should await a collection-scoped wiki rebuild."""
    collection_id = UUID("00000000-0000-0000-0000-000000000001")
    rebuild_calls: list[str] = []

    class FakeCollection:
        def __init__(self, collection_id: str) -> None:
            self.collection_id = collection_id

        async def delete(self, **kwargs: object) -> int:
            assert kwargs == {"file_id": "file-a"}
            return 2

    async def fake_rebuild(collection_id: str, **kwargs: object) -> SimpleNamespace:
        rebuild_calls.append(collection_id)
        return SimpleNamespace(model_dump=lambda mode="json": {})

    monkeypatch.setattr(documents_api, "Collection", FakeCollection)
    monkeypatch.setattr(documents_api, "rebuild_llm_wiki", fake_rebuild)

    response = await documents_api.documents_delete(
        collection_id=collection_id,
        document_id="file-a",
        delete_by="file_id",
    )

    assert response == {"success": True}
    assert rebuild_calls == [str(collection_id)]


async def test_documents_delete_skips_rebuild_after_noop_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-delete no-ops should preserve success but skip rebuild."""
    collection_id = UUID("00000000-0000-0000-0000-000000000001")
    rebuild_calls: list[str] = []

    class FakeCollection:
        def __init__(self, collection_id: str) -> None:
            self.collection_id = collection_id

        async def delete(self, **kwargs: object) -> int:
            return 0

    async def fake_rebuild(collection_id: str, **kwargs: object) -> SimpleNamespace:
        rebuild_calls.append(collection_id)
        return SimpleNamespace(model_dump=lambda mode="json": {})

    monkeypatch.setattr(documents_api, "Collection", FakeCollection)
    monkeypatch.setattr(documents_api, "rebuild_llm_wiki", fake_rebuild)

    response = await documents_api.documents_delete(
        collection_id=collection_id,
        document_id="missing-doc",
    )

    assert response == {"success": True}
    assert rebuild_calls == []


async def test_documents_delete_rejects_invalid_delete_by_before_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid delete_by values should fail before delete or rebuild."""
    collection_id = UUID("00000000-0000-0000-0000-000000000001")
    events: list[str] = []

    class FakeCollection:
        def __init__(self, collection_id: str) -> None:
            self.collection_id = collection_id

        async def delete(self, **kwargs: object) -> int:
            events.append("deleted")
            return 1

    async def fake_rebuild(collection_id: str, **kwargs: object) -> SimpleNamespace:
        events.append("rebuilt")
        return SimpleNamespace(model_dump=lambda mode="json": {})

    monkeypatch.setattr(documents_api, "Collection", FakeCollection)
    monkeypatch.setattr(documents_api, "rebuild_llm_wiki", fake_rebuild)

    with pytest.raises(HTTPException) as exc_info:
        await documents_api.documents_delete(
            collection_id=collection_id,
            document_id="doc-a",
            delete_by="garbage",
        )

    assert exc_info.value.status_code == 400
    assert events == []


async def test_documents_delete_rejects_invalid_delete_by_over_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid delete_by should be the endpoint's HTTP 400, not FastAPI 422."""
    collection_id = UUID("00000000-0000-0000-0000-000000000123")

    class FailingCollection:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("Collection should not be constructed")

    monkeypatch.setattr(documents_api, "Collection", FailingCollection)
    monkeypatch.setattr(
        documents_api,
        "rebuild_llm_wiki",
        lambda *_args, **_kwargs: pytest.fail("rebuild should not be called"),
    )

    from langconnect.server import APP

    transport = ASGITransport(app=APP, raise_app_exceptions=True)
    async with AsyncClient(base_url="http://test", transport=transport) as client:
        response = await client.delete(
            f"/collections/{collection_id}/documents/doc-1?delete_by=garbage"
        )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "delete_by must be either 'document_id' or 'file_id'."
    )


async def test_documents_delete_missing_collection_does_not_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing collection errors should propagate before rebuild."""
    collection_id = UUID("00000000-0000-0000-0000-000000000001")
    rebuild_calls: list[str] = []

    class FakeCollection:
        def __init__(self, collection_id: str) -> None:
            self.collection_id = collection_id

        async def delete(self, **kwargs: object) -> int:
            raise HTTPException(status_code=404, detail="Collection not found")

    async def fake_rebuild(collection_id: str, **kwargs: object) -> SimpleNamespace:
        rebuild_calls.append(collection_id)
        return SimpleNamespace(model_dump=lambda mode="json": {})

    monkeypatch.setattr(documents_api, "Collection", FakeCollection)
    monkeypatch.setattr(documents_api, "rebuild_llm_wiki", fake_rebuild)

    with pytest.raises(HTTPException) as exc_info:
        await documents_api.documents_delete(
            collection_id=collection_id, document_id="doc-a"
        )

    assert exc_info.value.status_code == 404
    assert rebuild_calls == []


async def test_documents_bulk_delete_rebuilds_once_for_mixed_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed bulk deletes should trigger exactly one rebuild."""
    collection_id = UUID("00000000-0000-0000-0000-000000000001")
    events: list[str] = []
    rebuild_calls: list[str] = []

    class FakeCollection:
        def __init__(self, collection_id: str) -> None:
            self.collection_id = collection_id

        async def delete_many(self, **kwargs: object) -> int:
            assert kwargs == {"document_ids": ["doc-a"], "file_ids": ["file-a"]}
            events.append("deleted")
            return 3

    async def fake_rebuild(collection_id: str, **kwargs: object) -> SimpleNamespace:
        rebuild_calls.append(collection_id)
        events.append("rebuilt")
        return SimpleNamespace(model_dump=lambda mode="json": {})

    monkeypatch.setattr(documents_api, "Collection", FakeCollection)
    monkeypatch.setattr(documents_api, "rebuild_llm_wiki", fake_rebuild)

    response = await documents_api.documents_bulk_delete(
        collection_id=collection_id,
        delete_request=documents_api.DocumentDelete(
            document_ids=["doc-a"],
            file_ids=["file-a"],
        ),
    )

    assert response == {"success": True, "deleted_count": 3}
    assert rebuild_calls == [str(collection_id)]
    assert events == ["deleted", "rebuilt"]


async def test_documents_bulk_delete_skips_rebuild_when_no_rows_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-op bulk deletes should not rebuild wiki artifacts."""
    collection_id = UUID("00000000-0000-0000-0000-000000000001")
    rebuild_calls: list[str] = []

    class FakeCollection:
        def __init__(self, collection_id: str) -> None:
            self.collection_id = collection_id

        async def delete_many(self, **kwargs: object) -> int:
            return 0

    async def fake_rebuild(collection_id: str, **kwargs: object) -> SimpleNamespace:
        rebuild_calls.append(collection_id)
        return SimpleNamespace(model_dump=lambda mode="json": {})

    monkeypatch.setattr(documents_api, "Collection", FakeCollection)
    monkeypatch.setattr(documents_api, "rebuild_llm_wiki", fake_rebuild)

    response = await documents_api.documents_bulk_delete(
        collection_id=collection_id,
        delete_request=documents_api.DocumentDelete(document_ids=["missing-doc"]),
    )

    assert response == {"success": True, "deleted_count": 0}
    assert rebuild_calls == []


async def test_documents_bulk_delete_validation_error_does_not_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bulk-delete validation errors should happen before rebuild."""
    collection_id = UUID("00000000-0000-0000-0000-000000000001")
    events: list[str] = []

    class FakeCollection:
        def __init__(self, collection_id: str) -> None:
            self.collection_id = collection_id

        async def delete_many(self, **kwargs: object) -> int:
            events.append("deleted")
            return 1

    async def fake_rebuild(collection_id: str, **kwargs: object) -> SimpleNamespace:
        events.append("rebuilt")
        return SimpleNamespace(model_dump=lambda mode="json": {})

    monkeypatch.setattr(documents_api, "Collection", FakeCollection)
    monkeypatch.setattr(documents_api, "rebuild_llm_wiki", fake_rebuild)

    with pytest.raises(HTTPException) as exc_info:
        await documents_api.documents_bulk_delete(
            collection_id=collection_id,
            delete_request=documents_api.DocumentDelete(),
        )

    assert exc_info.value.status_code == 400
    assert events == []


@pytest.mark.parametrize(
    "rebuild_error",
    [
        RuntimeError("secret path /tmp/internal prompt fragment"),
        OSError("permission denied /secret/path"),
        TimeoutError("provider timed out with token details"),
    ],
)
async def test_documents_delete_reports_sanitized_partial_success_when_rebuild_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    rebuild_error: Exception,
) -> None:
    """Delete rebuild failures should return sanitized partial-success details."""
    collection_id = UUID("00000000-0000-0000-0000-000000000001")

    class FakeCollection:
        def __init__(self, collection_id: str) -> None:
            self.collection_id = collection_id

        async def delete(self, **kwargs: object) -> int:
            return 1

    async def failing_rebuild(collection_id: str, **kwargs: object) -> None:
        raise rebuild_error

    monkeypatch.setattr(documents_api, "Collection", FakeCollection)
    monkeypatch.setattr(documents_api, "rebuild_llm_wiki", failing_rebuild)

    with (
        caplog.at_level("ERROR", logger=documents_api.logger.name),
        pytest.raises(HTTPException) as exc_info,
    ):
        await documents_api.documents_delete(
            collection_id=collection_id,
            document_id="doc-a",
        )

    assert exc_info.value.status_code == 500
    detail = exc_info.value.detail
    assert detail["success"] is False
    assert detail["error"] == "documents_deleted_wiki_rebuild_failed"
    assert detail["documents_deleted"] is True
    assert detail["deleted_count"] == 1
    assert detail["wiki_rebuild_error"] == "internal_error"
    assert detail["error_id"]
    assert "rebuild_llm_wiki" in detail["recovery"]
    assert "secret" not in str(detail)
    assert "/" + "tmp" not in str(detail)
    assert "provider timed out" not in str(detail)
    assert any(
        record.message == "llm_wiki_rebuild_failed_after_delete"
        and getattr(record, "collection_id", None) == str(collection_id)
        and getattr(record, "deleted_count", None) == 1
        and getattr(record, "error_id", None) == detail["error_id"]
        for record in caplog.records
    )


async def test_documents_bulk_delete_reports_sanitized_partial_success_when_rebuild_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bulk delete rebuild failures should include sanitized deleted_count detail."""
    collection_id = UUID("00000000-0000-0000-0000-000000000001")

    class FakeCollection:
        def __init__(self, collection_id: str) -> None:
            self.collection_id = collection_id

        async def delete_many(self, **kwargs: object) -> int:
            return 2

    async def failing_rebuild(collection_id: str, **kwargs: object) -> None:
        raise OSError("permission denied /secret/path")

    monkeypatch.setattr(documents_api, "Collection", FakeCollection)
    monkeypatch.setattr(documents_api, "rebuild_llm_wiki", failing_rebuild)

    with pytest.raises(HTTPException) as exc_info:
        await documents_api.documents_bulk_delete(
            collection_id=collection_id,
            delete_request=documents_api.DocumentDelete(
                document_ids=["doc-a", "doc-b"]
            ),
        )

    assert exc_info.value.status_code == 500
    detail = exc_info.value.detail
    assert detail["error"] == "documents_deleted_wiki_rebuild_failed"
    assert detail["deleted_count"] == 2
    assert detail["wiki_rebuild_error"] == "internal_error"
    assert "/secret/path" not in str(detail)


async def test_documents_delete_rebuild_cancelled_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelled rebuilds should propagate instead of becoming partial success."""
    collection_id = UUID("00000000-0000-0000-0000-000000000001")

    class FakeCollection:
        def __init__(self, collection_id: str) -> None:
            self.collection_id = collection_id

        async def delete(self, **kwargs: object) -> int:
            return 1

    async def cancelled_rebuild(collection_id: str, **kwargs: object) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(documents_api, "Collection", FakeCollection)
    monkeypatch.setattr(documents_api, "rebuild_llm_wiki", cancelled_rebuild)

    with pytest.raises(asyncio.CancelledError):
        await documents_api.documents_delete(
            collection_id=collection_id, document_id="doc-a"
        )


async def test_documents_create_and_list_and_delete_and_search() -> None:
    """Test creating, listing, deleting, and searching documents."""
    async with get_async_test_client() as client:
        # Create a collection for documents
        collection_name = "docs_test_col"
        col_payload = {"name": collection_name, "metadata": {"purpose": "doc-test"}}
        create_col = await client.post("/collections", json=col_payload)
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
        list_resp = await client.get(f"/collections/{collection_id}/documents")
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


async def test_documents_create_runs_llm_wiki_rebuild_after_successful_upsert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful uploads rebuild the collection LLM Wiki after vector upsert."""
    collection_id = UUID("00000000-0000-0000-0000-000000000001")
    rebuild_calls: list[str] = []

    async def fake_process_document(*args: object, **kwargs: object) -> list[Document]:
        return [Document(page_content="indexed text", metadata={"source": "doc.txt"})]

    class FakeCollection:
        def __init__(self, collection_id: str) -> None:
            self.collection_id = collection_id

        async def upsert(self, documents: list[Document]) -> list[str]:
            assert len(documents) == 1
            return ["chunk-a", "chunk-b"]

    async def fake_rebuild(collection_id: str, **kwargs: object) -> SimpleNamespace:
        rebuild_calls.append(collection_id)
        return SimpleNamespace(
            model_dump=lambda mode="json": {
                "collection_id": collection_id,
                "status": "rebuilt",
                "source_page_count": 1,
                "concept_page_count": 1,
                "page_count": 2,
                "chunk_count": 2,
                "pack_path": f"llm_wiki/collections/{collection_id}.json",
            }
        )

    monkeypatch.setattr(documents_api, "process_document", fake_process_document)
    monkeypatch.setattr(documents_api, "Collection", FakeCollection)
    monkeypatch.setattr(documents_api, "rebuild_llm_wiki", fake_rebuild)

    response = await documents_api.documents_create(
        collection_id=collection_id,
        files=[_upload_file()],
        metadatas_json=None,
        chunk_size=1000,
        chunk_overlap=200,
    )

    assert rebuild_calls == [str(collection_id)]
    assert response["success"] is True
    assert response["added_chunk_ids"] == ["chunk-a", "chunk-b"]
    assert response["llm_wiki"]["status"] == "rebuilt"
    assert response["llm_wiki"]["pack_path"].endswith(f"{collection_id}.json")


async def test_documents_create_reports_partial_success_when_rebuild_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vector upsert is not rolled back when upload-triggered rebuild fails."""
    collection_id = UUID("00000000-0000-0000-0000-000000000001")

    async def fake_process_document(*args: object, **kwargs: object) -> list[Document]:
        return [Document(page_content="indexed text", metadata={"source": "doc.txt"})]

    class FakeCollection:
        def __init__(self, collection_id: str) -> None:
            self.collection_id = collection_id

        async def upsert(self, documents: list[Document]) -> list[str]:
            return ["chunk-a"]

    async def failing_rebuild(collection_id: str, **kwargs: object) -> None:
        raise RuntimeError("wiki exploded")

    monkeypatch.setattr(documents_api, "process_document", fake_process_document)
    monkeypatch.setattr(documents_api, "Collection", FakeCollection)
    monkeypatch.setattr(documents_api, "rebuild_llm_wiki", failing_rebuild)

    with pytest.raises(HTTPException) as exc_info:
        await documents_api.documents_create(
            collection_id=collection_id,
            files=[_upload_file()],
            metadatas_json=None,
            chunk_size=1000,
            chunk_overlap=200,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == {
        "success": False,
        "error": "documents_indexed_wiki_rebuild_failed",
        "message": "Documents were indexed, but LLM Wiki rebuild failed.",
        "documents_indexed": True,
        "added_chunk_ids": ["chunk-a"],
        "wiki_rebuild_error": "wiki exploded",
    }


async def test_semantic_search_rejects_default_low_score_no_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default semantic search should not return unrelated nearest neighbors."""
    _patch_collection_details(monkeypatch)
    store = _FakeVectorStore(
        [
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
        ]
    )
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
    store = _FakeVectorStore(
        [
            (
                _doc(
                    "nonsense-boundary",
                    "Nearest-neighbor text unrelated to the query.",
                    source="distractor.pdf",
                ),
                0.4925,  # similarity ~= 0.67
            ),
        ]
    )
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
    store = _FakeVectorStore(
        [
            (
                _doc(
                    "low-score-match",
                    "A weak but caller-accepted semantic match.",
                    source="weak.pdf",
                ),
                0.54,
            ),
        ]
    )
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
    store = _FakeVectorStore(
        [
            (
                _doc(
                    "boundary-match",
                    "A caller-accepted lower confidence semantic match.",
                    source="weak.pdf",
                ),
                0.4925,  # similarity ~= 0.67
            ),
        ]
    )
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
    store = _FakeVectorStore(
        [
            (
                _doc(
                    "irrelevant-semantic",
                    "Unrelated nearest-neighbor content.",
                    source="distractor.pdf",
                ),
                0.54,
            ),
        ]
    )
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
    _patch_keyword_rows(
        monkeypatch,
        [
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
        ],
    )

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
    _patch_keyword_rows(
        monkeypatch,
        [
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
        ],
    )

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
        response = await client.get(f"/collections/{no_such_collection}/documents")
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
        list_resp = await client.get(
            f"/collections/{collection_id}/documents?limit=100"
        )
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
        list_resp_after_doc_delete = await client.get(
            f"/collections/{collection_id}/documents"
        )
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
        list_resp_after_file_delete = await client.get(
            f"/collections/{collection_id}/documents"
        )
        assert list_resp_after_file_delete.json() == []
