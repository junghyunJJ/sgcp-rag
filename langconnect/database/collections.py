"""Module defines CollectionManager and Collection classes.

1. CollectionManager: for managing collections of documents in a database.
2. Collection: for managing the contents of a specific collection.

The current implementations are based on langchain-postgres PGVector class.

Replace with your own implementation or favorite vectorstore if needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Mapping
from typing import Any, Literal, NotRequired, Optional, TypedDict

from fastapi import status
from fastapi.exceptions import HTTPException
from langchain_core.documents import Document

from langconnect.database.connection import get_db_connection, get_vectorstore

logger = logging.getLogger(__name__)

DEFAULT_SEMANTIC_MIN_SCORE = 0.68
HYBRID_FETCH_MULTIPLIER = 4
HYBRID_MIN_FETCH_K = 20
HYBRID_MAX_FETCH_K = 100
MAX_SEARCH_LIMIT = 100
HYBRID_SEMANTIC_WEIGHT = 0.7
HYBRID_KEYWORD_WEIGHT = 0.3

MetadataScalar = str | int | float | bool | None


def _distance_to_similarity(distance: float) -> float:
    """Convert PGVector distance to a bounded similarity score."""
    return 1 / (1 + distance)


def _raise_unsupported_filter(reason: str) -> None:
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unsupported metadata filter: {reason}",
    )


def _validate_metadata_filter(
    metadata_filter: Optional[dict[str, Any]],
) -> dict[str, MetadataScalar] | None:
    """Validate and flatten the supported metadata filter contract.

    Supported shape is simple scalar equality, optionally grouped under `$and`.
    This keeps semantic PGVector filters and keyword JSONB predicates aligned.
    """
    if metadata_filter is None:
        return None
    if not isinstance(metadata_filter, dict):
        _raise_unsupported_filter("filter must be an object")
    if not metadata_filter:
        return None

    flattened: dict[str, MetadataScalar] = {}

    def add_condition(key: str, value: object) -> None:
        if not isinstance(key, str) or not key:
            _raise_unsupported_filter("filter field names must be non-empty strings")
        if key.startswith("$"):
            _raise_unsupported_filter(f"operator {key!r} is not supported")
        if not key.isidentifier():
            _raise_unsupported_filter("filter field names must be valid identifiers")
        if isinstance(value, (dict, list)):
            _raise_unsupported_filter(f"field {key!r} must use scalar equality")
        if not isinstance(value, (str, int, float, bool, type(None))):
            _raise_unsupported_filter(f"field {key!r} must use scalar equality")
        if key in flattened and flattened[key] != value:
            _raise_unsupported_filter(f"conflicting values for field {key!r}")
        flattened[key] = value

    def visit(filter_part: dict[str, Any]) -> None:
        for key, value in filter_part.items():
            if key == "$and":
                if not isinstance(value, list):
                    _raise_unsupported_filter("$and must be a list of objects")
                for item in value:
                    if not isinstance(item, dict):
                        _raise_unsupported_filter("$and entries must be objects")
                    visit(item)
            else:
                add_condition(key, value)

    visit(metadata_filter)
    return flattened or None


def _metadata_from_row(row: Mapping[str, object]) -> dict[str, Any]:
    metadata = row["metadata"]
    if not metadata:
        return {}
    if isinstance(metadata, str):
        return json.loads(metadata)
    if isinstance(metadata, Mapping):
        return dict(metadata)
    return dict(metadata)


class CollectionDetails(TypedDict):
    """TypedDict for collection details."""

    uuid: str
    name: str
    metadata: dict[str, Any]
    # Temporary field used internally to workaround an issue with PGVector
    table_id: NotRequired[str]


class CollectionsManager:
    """Use to create, delete, update, and list document collections."""

    @staticmethod
    async def setup() -> None:
        """Set up method should run any necessary initialization code.

        For example, it could run SQL migrations to create the necessary tables.
        """
        logger.info("Starting database initialization...")
        get_vectorstore()
        logger.info("Database initialization complete.")

    async def list(
        self,
    ) -> list[CollectionDetails]:
        """List all collections ordered by logical name."""
        async with get_db_connection() as conn:
            records = await conn.fetch(
                """
                SELECT
                    c.uuid,
                    c.cmetadata,
                    COUNT(DISTINCT e.cmetadata->>'file_id') AS document_count,
                    COUNT(e.id) AS chunk_count
                FROM langchain_pg_collection c
                LEFT JOIN langchain_pg_embedding e ON c.uuid = e.collection_id
                GROUP BY c.uuid
                ORDER BY c.cmetadata->>'name';
                """
            )

        result: list[CollectionDetails] = []
        for r in records:
            metadata = json.loads(r["cmetadata"]) if r["cmetadata"] else {}
            name = metadata.pop("name", "Unnamed") if metadata else "Unnamed"
            result.append(
                {
                    "uuid": str(r["uuid"]),
                    "name": name,
                    "metadata": metadata or {},
                    "document_count": r["document_count"],
                    "chunk_count": r["chunk_count"],
                }
            )
        return result

    async def get(
        self,
        collection_id: str,
    ) -> CollectionDetails | None:
        """Fetch a single collection by UUID."""
        async with get_db_connection() as conn:
            rec = await conn.fetchrow(
                """
                SELECT
                    c.uuid,
                    c.name,
                    c.cmetadata,
                    COUNT(DISTINCT e.cmetadata->>'file_id') AS document_count,
                    COUNT(e.id) AS chunk_count
                  FROM langchain_pg_collection c
                  LEFT JOIN langchain_pg_embedding e ON c.uuid = e.collection_id
                 WHERE c.uuid = $1
                 GROUP BY c.uuid;
                """,
                collection_id,
            )

        if not rec:
            return None

        metadata = json.loads(rec["cmetadata"]) if rec["cmetadata"] else {}
        name = metadata.pop("name", "Unnamed") if metadata else "Unnamed"
        return {
            "uuid": str(rec["uuid"]),
            "name": name,
            "metadata": metadata or {},
            "document_count": rec["document_count"],
            "chunk_count": rec["chunk_count"],
            "table_id": rec["name"],
        }

    async def create(
        self,
        collection_name: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> CollectionDetails | None:
        """Create a new collection.

        Args:
            collection_name: The name of the new collection.
            metadata: Optional metadata for the collection.

        Returns:
            Details of the created collection or None if creation failed.
        """
        # check for existing name
        metadata = metadata.copy() if metadata else {}
        metadata["name"] = collection_name

        # For now assign a table identifier safe for SQL naming
        # Use hex string and prefix to avoid leading digits/hyphens
        table_id = f"tbl_{uuid.uuid4().hex}"

        # triggers PGVector to create both the vectorstore and DB entry
        get_vectorstore(table_id, collection_metadata=metadata)

        # Fetch the newly created table.
        async with get_db_connection() as conn:
            rec = await conn.fetchrow(
                """
                SELECT uuid, name, cmetadata
                  FROM langchain_pg_collection
                 WHERE name = $1;
                """,
                table_id,
            )
        if not rec:
            return None
        metadata = json.loads(rec["cmetadata"])
        name = metadata.pop("name")
        return {"uuid": str(rec["uuid"]), "name": name, "metadata": metadata}

    async def update(
        self,
        collection_id: str,
        *,
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> CollectionDetails:
        """Update collection metadata.

        Four cases:

        1) metadata only          → merge in metadata, keep old JSON->'name'
        2) metadata + new name    → merge metadata (including new 'name')
        3) new name only          → jsonb_set the 'name' key
        4) neither                → no-op, just fetch & return
        """
        # Case 4: no-op
        if metadata is None and name is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must update at least 1 attribute.",
            )

        # Case 1 & 2: metadata supplied (with or without new name)
        if metadata is not None:
            merged = metadata.copy()

            if name is not None:
                merged["name"] = name
            else:
                # pull existing friendly name so we don't lose it
                existing = await self.get(collection_id)
                if not existing:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Collection '{collection_id}' not found.",
                    )
                merged["name"] = existing["name"]

            metadata_json = json.dumps(merged)

            async with get_db_connection() as conn:
                rec = await conn.fetchrow(
                    """
                    UPDATE langchain_pg_collection
                       SET cmetadata = $1::jsonb
                     WHERE uuid = $2
                    RETURNING uuid, cmetadata;
                    """,
                    metadata_json,
                    collection_id,
                )

        # Case 3: name only
        else:  # metadata is None but name is not None
            async with get_db_connection() as conn:
                rec = await conn.fetchrow(
                    """
                    UPDATE langchain_pg_collection
                       SET cmetadata = jsonb_set(
                             cmetadata::jsonb,
                             '{name}',
                             to_jsonb($1::text),
                             true
                           )
                     WHERE uuid = $2
                    RETURNING uuid, cmetadata;
                    """,
                    name,
                    collection_id,
                )

        if not rec:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{collection_id}' not found.",
            )

        full_meta = json.loads(rec["cmetadata"])
        friendly_name = full_meta.pop("name", "Unnamed")

        return {
            "uuid": str(rec["uuid"]),
            "name": friendly_name,
            "metadata": full_meta,
        }

    async def delete(
        self,
        collection_id: str,
    ) -> int:
        """Delete a collection by UUID.
        Returns number of rows deleted (1).
        Raises 404 if no such collection.
        """
        async with get_db_connection() as conn:
            result = await conn.execute(
                """
                DELETE FROM langchain_pg_collection
                 WHERE uuid = $1;
                """,
                collection_id,
            )
        return int(result.split()[-1])


class Collection:
    """A collection of documents.

    Use to add, delete, list, and search documents to a given collection.
    """

    def __init__(self, collection_id: str) -> None:
        """Initialize the collection by collection ID."""
        self.collection_id = collection_id

    async def _get_details_or_raise(self) -> dict[str, Any]:
        """Get collection details if it exists, otherwise raise an error."""
        details = await CollectionsManager().get(self.collection_id)
        if not details:
            raise HTTPException(status_code=404, detail="Collection not found")
        return details

    async def upsert(self, documents: list[Document]) -> list[str]:
        """Add one or more documents to the collection."""
        details = await self._get_details_or_raise()
        store = get_vectorstore(collection_name=details["table_id"])
        added_ids = await asyncio.to_thread(store.add_documents, documents)
        return added_ids

    async def delete(
        self,
        *,
        file_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> int:
        """Delete embeddings by file id or individual document id.

        Args:
            file_id: Deletes all chunks from a specific file
            document_id: Deletes a specific chunk/document

        Returns:
            Number of deleted embeddings. Returns 0 when the collection exists but
            no matching document or file chunks were found.
        """
        async with get_db_connection() as conn:
            if document_id:
                # Delete specific document by ID
                result = await conn.execute(
                    """
                    DELETE FROM langchain_pg_embedding AS lpe
                    USING langchain_pg_collection AS lpc
                    WHERE lpe.collection_id = lpc.uuid
                      AND lpc.uuid = $1
                      AND lpe.id = $2
                    """,
                    self.collection_id,
                    document_id,
                )
                deleted_count = int(result.split()[-1])
                logger.info(
                    f"Deleted {deleted_count} document with id {document_id!r}."
                )
            elif file_id:
                # Delete all documents from a file
                result = await conn.execute(
                    """
                    DELETE FROM langchain_pg_embedding AS lpe
                    USING langchain_pg_collection AS lpc
                    WHERE lpe.collection_id = lpc.uuid
                      AND lpc.uuid = $1
                      AND lpe.cmetadata->>'file_id' = $2
                    """,
                    self.collection_id,
                    file_id,
                )
                deleted_count = int(result.split()[-1])
                logger.info(f"Deleted {deleted_count} embeddings for file {file_id!r}.")
            else:
                raise ValueError("Either file_id or document_id must be provided")

            # For now if deleted count is 0, let's verify that the collection exists.
            if deleted_count == 0:
                await self._get_details_or_raise()
        return deleted_count

    async def delete_many(
        self,
        *,
        document_ids: Optional[list[str]] = None,
        file_ids: Optional[list[str]] = None,
    ) -> int:
        """Delete multiple documents by a list of document IDs or file IDs."""
        if not document_ids and not file_ids:
            raise ValueError("Either document_ids or file_ids must be provided.")

        deleted_count = 0
        async with get_db_connection() as conn:
            if document_ids:
                result = await conn.execute(
                    """
                    DELETE FROM langchain_pg_embedding AS lpe
                    USING langchain_pg_collection AS lpc
                    WHERE lpe.collection_id = lpc.uuid
                      AND lpc.uuid = $1
                      AND lpe.id = ANY($2::text[])
                    """,
                    self.collection_id,
                    document_ids,
                )
                deleted_count += int(result.split()[-1])

            if file_ids:
                result = await conn.execute(
                    """
                    DELETE FROM langchain_pg_embedding AS lpe
                    USING langchain_pg_collection AS lpc
                    WHERE lpe.collection_id = lpc.uuid
                      AND lpc.uuid = $1
                      AND lpe.cmetadata->>'file_id' = ANY($2::text[])
                    """,
                    self.collection_id,
                    file_ids,
                )
                deleted_count += int(result.split()[-1])

        return deleted_count

    async def list(self, *, limit: int = 10, offset: int = 0) -> list[dict[str, Any]]:
        """List all document chunks in this collection."""
        async with get_db_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT lpe.id,
                       lpe.document,
                       lpe.cmetadata
                  FROM langchain_pg_embedding lpe
                  JOIN langchain_pg_collection lpc
                    ON lpe.collection_id = lpc.uuid
                 WHERE lpc.uuid = $1
                 ORDER BY lpe.cmetadata->>'file_id', lpe.id
                 LIMIT  $2
                OFFSET $3
                """,
                self.collection_id,
                limit,
                offset,
            )

        docs: list[dict[str, Any]] = []
        for r in rows:
            metadata = json.loads(r["cmetadata"]) if r["cmetadata"] else {}
            docs.append(
                {
                    "id": str(r["id"]),
                    "content": r["document"],
                    "metadata": metadata or {},
                    "collection_id": str(self.collection_id),
                    # For compatibility with UI expecting 'page_content'
                    "page_content": r["document"],
                }
            )

        if not docs:
            # For now, if no documents, let's check that the collection exists.
            # It may make sense to consider this a 200 OK with empty list.
            # And make sure its user responsibility to check that the collection
            # exists.
            await self._get_details_or_raise()
        return docs

    async def get(self, document_id: str) -> dict[str, Any]:
        """Fetch a single chunk by its document id within this collection."""
        async with get_db_connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT e.id, e.document, e.cmetadata
                  FROM langchain_pg_embedding e
                  JOIN langchain_pg_collection c
                    ON e.collection_id = c.uuid
                 WHERE e.id = $1
                   AND c.uuid = $2
                """,
                document_id,
                self.collection_id,
            )
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        metadata = json.loads(row["cmetadata"]) if row["cmetadata"] else {}
        return {
            "id": str(row["id"]),
            "content": row["document"],
            "page_content": row["document"],
            "metadata": metadata,
            "collection_id": str(self.collection_id),
        }

    async def get_many_by_source_refs(
        self,
        source_refs: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Fetch chunks by wiki source-ref file/chunk pairs in ref order."""
        refs = [
            {"file_id": ref["file_id"].strip(), "chunk_id": ref["chunk_id"].strip()}
            for ref in source_refs
            if isinstance(ref, dict)
            and isinstance(ref.get("file_id"), str)
            and isinstance(ref.get("chunk_id"), str)
            and ref["file_id"].strip()
            and ref["chunk_id"].strip()
        ]
        if not refs:
            return []

        file_ids = [ref["file_id"] for ref in refs]
        chunk_ids = [ref["chunk_id"] for ref in refs]
        async with get_db_connection() as conn:
            rows = await conn.fetch(
                """
                WITH refs AS (
                    SELECT file_id, chunk_id, ord
                      FROM unnest($2::text[], $3::text[])
                           WITH ORDINALITY AS t(file_id, chunk_id, ord)
                )
                SELECT refs.file_id AS wiki_file_id,
                       refs.chunk_id AS wiki_chunk_id,
                       e.id AS id,
                       e.document AS page_content,
                       e.cmetadata AS metadata
                  FROM refs
                  JOIN langchain_pg_embedding e
                    ON e.id::text = refs.chunk_id
                  JOIN langchain_pg_collection c
                    ON e.collection_id = c.uuid
                 WHERE c.uuid = $1
                   AND e.cmetadata->>'file_id' = refs.file_id
                 ORDER BY refs.ord
                """,
                self.collection_id,
                file_ids,
                chunk_ids,
            )

        docs: list[dict[str, Any]] = []
        for row in rows:
            metadata = _metadata_from_row(row)
            try:
                wiki_file_id = str(row["wiki_file_id"])
            except (IndexError, KeyError):
                wiki_file_id = str(metadata.get("file_id", ""))
            try:
                wiki_chunk_id = str(row["wiki_chunk_id"])
            except (IndexError, KeyError):
                wiki_chunk_id = str(row["id"])
            metadata.update(
                {
                    "wiki_promoted": True,
                    "wiki_file_id": wiki_file_id,
                    "wiki_chunk_id": wiki_chunk_id,
                }
            )
            docs.append(
                {
                    "id": str(row["id"]),
                    "page_content": row["page_content"],
                    "content": row["page_content"],
                    "metadata": metadata,
                    "score": 1.0,
                }
            )
        return docs

    async def search(
        self,
        query: str,
        *,
        limit: int = 4,
        search_type: Literal["semantic", "keyword", "hybrid"] = "semantic",
        filter: Optional[dict[str, Any]] = None,  # noqa: A002
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Run a search in the collection.

        Args:
            query: The search query string
            limit: Maximum number of results to return
            search_type: Type of search - "semantic", "keyword", or "hybrid"
            filter: Optional metadata filter to apply to results
            min_score: Optional semantic similarity threshold from 0 to 1

        Returns:
            List of search results with id, page_content, metadata, and score
        """
        if search_type not in ["semantic", "keyword", "hybrid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid search type: {search_type}. Must be 'semantic', 'keyword', or 'hybrid'.",
            )

        if not 1 <= limit <= MAX_SEARCH_LIMIT:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="limit must be between 1 and 100.",
            )

        if min_score is not None and not 0 <= min_score <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="min_score must be between 0 and 1.",
            )

        metadata_filter = _validate_metadata_filter(filter)
        semantic_min_score = (
            min_score if min_score is not None else DEFAULT_SEMANTIC_MIN_SCORE
        )
        details = await self._get_details_or_raise()

        if search_type == "semantic":
            store = get_vectorstore(collection_name=details["table_id"])
            results = await asyncio.to_thread(
                store.similarity_search_with_score,
                query,
                k=limit,
                filter=metadata_filter,
            )

            formatted_results = [
                {
                    "id": doc.id,
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": _distance_to_similarity(distance),
                }
                for doc, distance in results
                if _distance_to_similarity(distance) >= semantic_min_score
            ]
            return formatted_results[:limit]

        if search_type == "keyword":
            async with get_db_connection() as conn:
                if metadata_filter:
                    rows = await conn.fetch(
                        """
                        SELECT e.id as id,
                               e.document as page_content,
                               e.cmetadata as metadata,
                               ts_rank(to_tsvector('english', e.document),
                                      plainto_tsquery('english', $1)) as score
                        FROM langchain_pg_embedding e
                        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                        WHERE c.uuid = $2
                          AND to_tsvector('english', e.document) @@ plainto_tsquery('english', $1)
                          AND e.cmetadata::jsonb @> $3::jsonb
                        ORDER BY score DESC
                        LIMIT $4
                        """,
                        query,
                        self.collection_id,
                        json.dumps(metadata_filter),
                        limit,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT e.id as id,
                               e.document as page_content,
                               e.cmetadata as metadata,
                               ts_rank(to_tsvector('english', e.document),
                                      plainto_tsquery('english', $1)) as score
                        FROM langchain_pg_embedding e
                        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                        WHERE c.uuid = $2
                          AND to_tsvector('english', e.document) @@ plainto_tsquery('english', $1)
                        ORDER BY score DESC
                        LIMIT $3
                        """,
                        query,
                        self.collection_id,
                        limit,
                    )

            return [
                {
                    "id": str(row["id"]),
                    "page_content": row["page_content"],
                    "metadata": _metadata_from_row(row),
                    "score": float(row["score"]),
                }
                for row in rows
            ][:limit]

        # hybrid
        fetch_k = min(
            max(limit * HYBRID_FETCH_MULTIPLIER, HYBRID_MIN_FETCH_K),
            HYBRID_MAX_FETCH_K,
        )
        store = get_vectorstore(collection_name=details["table_id"])
        semantic_results = await asyncio.to_thread(
            store.similarity_search_with_score,
            query,
            k=fetch_k,
            filter=metadata_filter,
        )

        async with get_db_connection() as conn:
            if metadata_filter:
                keyword_rows = await conn.fetch(
                    """
                    SELECT e.id as id,
                           e.document as page_content,
                           e.cmetadata as metadata,
                           ts_rank(to_tsvector('english', e.document),
                                  plainto_tsquery('english', $1)) as score
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.uuid = $2
                      AND to_tsvector('english', e.document) @@ plainto_tsquery('english', $1)
                      AND e.cmetadata::jsonb @> $3::jsonb
                    ORDER BY score DESC
                    LIMIT $4
                    """,
                    query,
                    self.collection_id,
                    json.dumps(metadata_filter),
                    fetch_k,
                )
            else:
                keyword_rows = await conn.fetch(
                    """
                    SELECT e.id as id,
                           e.document as page_content,
                           e.cmetadata as metadata,
                           ts_rank(to_tsvector('english', e.document),
                                  plainto_tsquery('english', $1)) as score
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.uuid = $2
                      AND to_tsvector('english', e.document) @@ plainto_tsquery('english', $1)
                    ORDER BY score DESC
                    LIMIT $3
                    """,
                    query,
                    self.collection_id,
                    fetch_k,
                )

        combined_results: dict[str, dict[str, Any]] = {}

        for doc, distance in semantic_results:
            similarity_score = _distance_to_similarity(distance)
            if similarity_score < semantic_min_score:
                continue
            doc_id = str(doc.id)
            combined_results[doc_id] = {
                "id": doc_id,
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "semantic_score": similarity_score,
                "keyword_score": 0.0,
            }

        if keyword_rows:
            max_keyword_score = max(
                (float(row["score"]) for row in keyword_rows), default=1.0
            )
            for row in keyword_rows:
                doc_id = str(row["id"])
                normalized_score = (
                    float(row["score"]) / max_keyword_score
                    if max_keyword_score > 0
                    else 0
                )
                if doc_id in combined_results:
                    combined_results[doc_id]["keyword_score"] = normalized_score
                else:
                    combined_results[doc_id] = {
                        "id": doc_id,
                        "page_content": row["page_content"],
                        "metadata": _metadata_from_row(row),
                        "semantic_score": 0.0,
                        "keyword_score": normalized_score,
                    }

        fused_results = []
        for result in combined_results.values():
            semantic_score = float(result["semantic_score"])
            keyword_score = float(result["keyword_score"])
            fused_score = (
                semantic_score * HYBRID_SEMANTIC_WEIGHT
                + keyword_score * HYBRID_KEYWORD_WEIGHT
            )
            fused_results.append(
                {
                    "id": result["id"],
                    "page_content": result["page_content"],
                    "metadata": result["metadata"],
                    "score": fused_score,
                }
            )

        return sorted(fused_results, key=lambda x: x["score"], reverse=True)[:limit]
