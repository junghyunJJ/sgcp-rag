"""Module defines CollectionManager and Collection classes.

1. CollectionManager: for managing collections of documents in a database.
2. Collection: for managing the contents of a specific collection.

The current implementations are based on langchain-postgres PGVector class.

Replace with your own implementation or favorite vectorstore if needed.
"""

import builtins
import json
import logging
import uuid
from typing import Any, Literal, NotRequired, Optional, TypedDict

from fastapi import status
from fastapi.exceptions import HTTPException
from langchain_core.documents import Document

from langconnect.database.connection import get_db_connection, get_vectorstore

logger = logging.getLogger(__name__)


class CollectionDetails(TypedDict):
    """TypedDict for collection details."""

    uuid: str
    name: str
    metadata: dict[str, Any]
    # Temporary field used internally to workaround an issue with PGVector
    table_id: NotRequired[str]


class CollectionsManager:
    """Use to create, delete, update, and list document collections."""

    def __init__(self, user_id: str | None) -> None:
        """Initialize the collection manager with an optional user ID."""
        self.user_id = user_id

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
        """List all collections, optionally filtered by user, ordered by logical name."""
        async with get_db_connection() as conn:
            if self.user_id:
                records = await conn.fetch(
                    """
                    SELECT 
                        c.uuid, 
                        c.cmetadata,
                        COUNT(DISTINCT e.cmetadata->>'file_id') AS document_count,
                        COUNT(e.id) AS chunk_count
                    FROM langchain_pg_collection c
                    LEFT JOIN langchain_pg_embedding e ON c.uuid = e.collection_id
                    WHERE c.cmetadata->>'owner_id' = $1
                    GROUP BY c.uuid
                    ORDER BY c.cmetadata->>'name';
                    """,
                    self.user_id,
                )
            else:
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
        """Fetch a single collection by UUID, optionally checking ownership."""
        async with get_db_connection() as conn:
            if self.user_id:
                rec = await conn.fetchrow(
                    """
                    SELECT uuid, name, cmetadata
                      FROM langchain_pg_collection
                     WHERE uuid = $1
                       AND cmetadata->>'owner_id' = $2;
                    """,
                    collection_id,
                    self.user_id,
                )
            else:
                rec = await conn.fetchrow(
                    """
                    SELECT uuid, name, cmetadata
                      FROM langchain_pg_collection
                     WHERE uuid = $1;
                    """,
                    collection_id,
                )

        if not rec:
            return None

        metadata = json.loads(rec["cmetadata"])
        name = metadata.pop("name", "Unnamed")
        return {
            "uuid": str(rec["uuid"]),
            "name": name,
            "metadata": metadata,
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
        if self.user_id:
            metadata["owner_id"] = self.user_id
        metadata["name"] = collection_name

        # For now assign a table identifier safe for SQL naming
        # Use hex string and prefix to avoid leading digits/hyphens
        table_id = f"tbl_{uuid.uuid4().hex}"

        # triggers PGVector to create both the vectorstore and DB entry
        get_vectorstore(table_id, collection_metadata=metadata)

        # Fetch the newly created table.
        async with get_db_connection() as conn:
            if self.user_id:
                rec = await conn.fetchrow(
                    """
                    SELECT uuid, name, cmetadata
                      FROM langchain_pg_collection
                     WHERE name = $1
                       AND cmetadata->>'owner_id' = $2;
                    """,
                    table_id,
                    self.user_id,
                )
            else:
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
            # merge in owner_id (if exists) + optional new name
            merged = metadata.copy()
            if self.user_id:
                merged["owner_id"] = self.user_id

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
                if self.user_id:
                    rec = await conn.fetchrow(
                        """
                        UPDATE langchain_pg_collection
                           SET cmetadata = $1::jsonb
                         WHERE uuid = $2
                           AND cmetadata->>'owner_id' = $3
                        RETURNING uuid, cmetadata;
                        """,
                        metadata_json,
                        collection_id,
                        self.user_id,
                    )
                else:
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
                if self.user_id:
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
                           AND cmetadata->>'owner_id' = $3
                        RETURNING uuid, cmetadata;
                        """,
                        name,
                        collection_id,
                        self.user_id,
                    )
                else:
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
            if self.user_id:
                result = await conn.execute(
                    """
                    DELETE FROM langchain_pg_collection
                     WHERE uuid = $1
                       AND cmetadata->>'owner_id' = $2;
                    """,
                    collection_id,
                    self.user_id,
                )
            else:
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

    def __init__(self, collection_id: str, user_id: str | None) -> None:
        """Initialize the collection by collection ID with optional user ID."""
        self.collection_id = collection_id
        self.user_id = user_id

    async def _get_details_or_raise(self) -> dict[str, Any]:
        """Get collection details if it exists, otherwise raise an error."""
        details = await CollectionsManager(self.user_id).get(self.collection_id)
        if not details:
            raise HTTPException(status_code=404, detail="Collection not found")
        return details

    async def upsert(self, documents: list[Document]) -> list[str]:
        """Add one or more documents to the collection."""
        details = await self._get_details_or_raise()
        store = get_vectorstore(collection_name=details["table_id"])
        added_ids = store.add_documents(documents)
        return added_ids

    async def delete(
        self,
        *,
        file_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> bool:
        """Delete embeddings by file id or individual document id.

        Args:
            file_id: Deletes all chunks from a specific file
            document_id: Deletes a specific chunk/document
        """
        async with get_db_connection() as conn:
            if document_id:
                # Delete specific document by ID
                if self.user_id:
                    delete_sql = """
                        DELETE FROM langchain_pg_embedding AS lpe
                        USING langchain_pg_collection AS lpc
                        WHERE lpe.collection_id = lpc.uuid
                          AND lpc.uuid = $1
                          AND lpc.cmetadata->>'owner_id' = $2
                          AND lpe.id = $3
                    """
                    result = await conn.execute(
                        delete_sql,
                        self.collection_id,
                        self.user_id,
                        document_id,
                    )
                else:
                    delete_sql = """
                        DELETE FROM langchain_pg_embedding AS lpe
                        USING langchain_pg_collection AS lpc
                        WHERE lpe.collection_id = lpc.uuid
                          AND lpc.uuid = $1
                          AND lpe.id = $2
                    """
                    result = await conn.execute(
                        delete_sql,
                        self.collection_id,
                        document_id,
                    )
                deleted_count = int(result.split()[-1])
                logger.info(
                    f"Deleted {deleted_count} document with id {document_id!r}."
                )
            elif file_id:
                # Delete all documents from a file
                if self.user_id:
                    delete_sql = """
                        DELETE FROM langchain_pg_embedding AS lpe
                        USING langchain_pg_collection AS lpc
                        WHERE lpe.collection_id   = lpc.uuid
                          AND lpc.uuid             = $1
                          AND lpc.cmetadata->>'owner_id' = $2
                          AND lpe.cmetadata->>'file_id'   = $3
                    """
                    result = await conn.execute(
                        delete_sql,
                        self.collection_id,
                        self.user_id,
                        file_id,
                    )
                else:
                    delete_sql = """
                        DELETE FROM langchain_pg_embedding AS lpe
                        USING langchain_pg_collection AS lpc
                        WHERE lpe.collection_id   = lpc.uuid
                          AND lpc.uuid             = $1
                          AND lpe.cmetadata->>'file_id'   = $2
                    """
                    result = await conn.execute(
                        delete_sql,
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
        return True

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
                if self.user_id:
                    result = await conn.execute(
                        """
                        DELETE FROM langchain_pg_embedding AS lpe
                        USING langchain_pg_collection AS lpc
                        WHERE lpe.collection_id = lpc.uuid
                          AND lpc.uuid = $1
                          AND lpc.cmetadata->>'owner_id' = $2
                          AND lpe.id = ANY($3::text[])
                        """,
                        self.collection_id,
                        self.user_id,
                        document_ids,
                    )
                else:
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
                if self.user_id:
                    result = await conn.execute(
                        """
                        DELETE FROM langchain_pg_embedding AS lpe
                        USING langchain_pg_collection AS lpc
                        WHERE lpe.collection_id = lpc.uuid
                          AND lpc.uuid = $1
                          AND lpc.cmetadata->>'owner_id' = $2
                          AND lpe.cmetadata->>'file_id' = ANY($3::text[])
                        """,
                        self.collection_id,
                        self.user_id,
                        file_ids,
                    )
                else:
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
            if self.user_id:
                rows = await conn.fetch(
                    """
                    SELECT lpe.id,
                           lpe.document,
                           lpe.cmetadata
                      FROM langchain_pg_embedding lpe
                      JOIN langchain_pg_collection lpc
                        ON lpe.collection_id = lpc.uuid
                     WHERE lpc.uuid = $1
                       AND lpc.cmetadata->>'owner_id' = $2
                     ORDER BY lpe.cmetadata->>'file_id', lpe.id
                     LIMIT  $3
                    OFFSET $4
                    """,
                    self.collection_id,
                    self.user_id,
                    limit,
                    offset,
                )
            else:
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
        """Fetch a single chunk by its UUID, optionally verifying collection ownership."""
        async with get_db_connection() as conn:
            if self.user_id:
                row = await conn.fetchrow(
                    """
                    SELECT e.uuid, e.document, e.cmetadata
                      FROM langchain_pg_embedding e
                      JOIN langchain_pg_collection c
                        ON e.collection_id = c.uuid
                     WHERE e.uuid = $1
                       AND c.cmetadata->>'owner_id' = $2
                       AND c.uuid = $3
                    """,
                    document_id,
                    self.user_id,
                    self.collection_id,
                )
            else:
                row = await conn.fetchrow(
                    """
                    SELECT e.uuid, e.document, e.cmetadata
                      FROM langchain_pg_embedding e
                      JOIN langchain_pg_collection c
                        ON e.collection_id = c.uuid
                     WHERE e.uuid = $1
                       AND c.uuid = $2
                    """,
                    document_id,
                    self.collection_id,
                )
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        metadata = json.loads(row["cmetadata"]) if row["cmetadata"] else {}
        return {
            "id": str(row["uuid"]),
            "content": row["document"],
            "metadata": metadata,
        }

    async def search(
        self,
        query: str,
        *,
        limit: int = 4,
        search_type: Literal["semantic", "keyword", "hybrid"] = "semantic",
        filter: Optional[dict[str, Any]] = None,
    ) -> builtins.list[dict[str, Any]]:
        """Run a search in the collection.

        Args:
            query: The search query string
            limit: Maximum number of results to return
            search_type: Type of search - "semantic", "keyword", or "hybrid"
            filter: Optional metadata filter to apply to results

        Returns:
            List of search results with id, page_content, metadata, and score
        """
        if search_type not in ["semantic", "keyword", "hybrid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid search type: {search_type}. Must be 'semantic', 'keyword', or 'hybrid'.",
            )

        details = await self._get_details_or_raise()

        # Helper function to apply metadata filter
        def apply_metadata_filter(
            results: list[dict[str, Any]], filter_dict: dict[str, Any]
        ) -> list[dict[str, Any]]:
            """Apply metadata filter to search results."""
            if not filter_dict:
                return results

            filtered_results = []
            for result in results:
                result_metadata = result.get("metadata", {})
                # Check if all filter conditions are met
                match = True
                for key, value in filter_dict.items():
                    if key not in result_metadata or result_metadata[key] != value:
                        match = False
                        break
                if match:
                    filtered_results.append(result)
            return filtered_results

        if search_type == "semantic":
            # Current semantic search implementation
            store = get_vectorstore(collection_name=details["table_id"])
            # Get more results initially if filter is applied
            k = limit * 3 if filter else limit
            results = store.similarity_search_with_score(query, k=k)

            # Convert to standard format
            formatted_results = [
                {
                    "id": doc.id,
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score,
                }
                for doc, score in results
            ]

            # Apply metadata filter
            filtered_results = apply_metadata_filter(formatted_results, filter)

            # Return only the requested limit
            return filtered_results[:limit]

        if search_type == "keyword":
            # Full-text search using PostgreSQL
            # Get more results initially if filter is applied
            search_limit = limit * 3 if filter else limit

            async with get_db_connection() as conn:
                if self.user_id:
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
                          AND c.cmetadata->>'owner_id' = $3
                          AND to_tsvector('english', e.document) @@ plainto_tsquery('english', $1)
                        ORDER BY score DESC
                        LIMIT $4
                        """,
                        query,
                        self.collection_id,
                        self.user_id,
                        search_limit,
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
                        search_limit,
                    )

            formatted_results = [
                {
                    "id": str(row["id"]),
                    "page_content": row["page_content"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "score": float(row["score"]),
                }
                for row in rows
            ]

            # Apply metadata filter
            filtered_results = apply_metadata_filter(formatted_results, filter)

            # Return only the requested limit
            return filtered_results[:limit]

        # hybrid
        # Get semantic search results
        store = get_vectorstore(collection_name=details["table_id"])
        semantic_results = store.similarity_search_with_score(query, k=limit * 2)

        # Get keyword search results
        async with get_db_connection() as conn:
            if self.user_id:
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
                          AND c.cmetadata->>'owner_id' = $3
                          AND to_tsvector('english', e.document) @@ plainto_tsquery('english', $1)
                        ORDER BY score DESC
                        LIMIT $4
                        """,
                    query,
                    self.collection_id,
                    self.user_id,
                    limit * 2,
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
                    limit * 2,
                )

        # Combine and deduplicate results
        combined_results = {}

        # Add semantic results with normalized scores
        max_semantic_score = max(
            (score for _, score in semantic_results), default=1.0
        )
        for doc, score in semantic_results:
            normalized_score = (
                score / max_semantic_score if max_semantic_score > 0 else 0
            )
            combined_results[doc.id] = {
                "id": doc.id,
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "semantic_score": normalized_score,
                "keyword_score": 0,
                "combined_score": normalized_score * 0.7,  # 70% weight for semantic
            }

        # Add keyword results with normalized scores
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
                    # Document exists, update scores
                    combined_results[doc_id]["keyword_score"] = normalized_score
                    combined_results[doc_id]["combined_score"] = (
                        combined_results[doc_id]["semantic_score"] * 0.7
                        + normalized_score * 0.3  # 30% weight for keyword
                    )
                else:
                    # New document from keyword search
                    combined_results[doc_id] = {
                        "id": doc_id,
                        "page_content": row["page_content"],
                        "metadata": json.loads(row["metadata"])
                        if row["metadata"]
                        else {},
                        "semantic_score": 0,
                        "keyword_score": normalized_score,
                        "combined_score": normalized_score * 0.3,
                    }

        # Convert combined results to list format
        all_results = [
            {
                "id": result["id"],
                "page_content": result["page_content"],
                "metadata": result["metadata"],
                "score": result["combined_score"],
            }
            for result in combined_results.values()
        ]

        # Apply metadata filter
        filtered_results = apply_metadata_filter(all_results, filter)

        # Sort by combined score and return top results
        sorted_results = sorted(
            filtered_results, key=lambda x: x["score"], reverse=True
        )[:limit]

        return sorted_results
