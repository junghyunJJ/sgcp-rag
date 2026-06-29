import logging
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from langchain_core.documents import Document
from pydantic import TypeAdapter, ValidationError

from langconnect.database.collections import Collection
from langconnect.models import (
    DocumentDelete,
    DocumentResponse,
    SearchQuery,
    SearchResult,
)
from langconnect.services import process_document
from langconnect.services.llm_wiki import rebuild_llm_wiki

# Create a TypeAdapter that enforces “list of dict”
_metadata_adapter = TypeAdapter(list[dict[str, Any]])

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])


async def _rebuild_llm_wiki_after_delete_or_raise(
    collection_id: UUID,
    *,
    deleted_count: int,
) -> None:
    """Rebuild delete-mutated wiki artifacts or raise a partial-success HTTP 500.

    Raises:
        HTTPException: When deletion succeeded but the follow-up wiki rebuild failed.
        asyncio.CancelledError and other BaseException subclasses are intentionally
        not caught.
    """
    try:
        await rebuild_llm_wiki(str(collection_id))
    except Exception as wiki_exc:
        error_id = uuid4().hex
        logger.exception(
            "llm_wiki_rebuild_failed_after_delete",
            extra={
                "collection_id": str(collection_id),
                "deleted_count": deleted_count,
                "error_id": error_id,
            },
        )
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "documents_deleted_wiki_rebuild_failed",
                "message": "Documents were deleted, but LLM Wiki rebuild failed.",
                "documents_deleted": True,
                "deleted_count": deleted_count,
                "wiki_rebuild_error": "internal_error",
                "error_id": error_id,
                "recovery": "Call rebuild_llm_wiki(collection_id) to retry.",
            },
        ) from wiki_exc


@router.delete(
    "/collections/{collection_id}/documents",
    response_model=dict[str, Any],
)
async def documents_bulk_delete(
    collection_id: UUID,
    delete_request: DocumentDelete,
):
    """Deletes multiple documents from a collection by their IDs or file IDs."""
    if not delete_request.document_ids and not delete_request.file_ids:
        raise HTTPException(
            status_code=400,
            detail="Either document_ids or file_ids must be provided.",
        )

    collection = Collection(collection_id=str(collection_id))
    deleted_count = await collection.delete_many(
        document_ids=delete_request.document_ids,
        file_ids=delete_request.file_ids,
    )
    if deleted_count > 0:
        await _rebuild_llm_wiki_after_delete_or_raise(
            collection_id,
            deleted_count=deleted_count,
        )

    return {"success": True, "deleted_count": deleted_count}


MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


@router.post("/collections/{collection_id}/documents", response_model=dict[str, Any])
async def documents_create(
    collection_id: UUID,
    files: list[UploadFile] = File(...),
    metadatas_json: str | None = Form(None),
    chunk_size: int = Form(3000, ge=10, le=10000),
    chunk_overlap: int = Form(200, ge=0, le=5000),
    rebuild_wiki: bool = Form(True),
):
    """Processes and indexes (adds) new document files with optional metadata.

    Args:
        collection_id: UUID of the collection to add documents to
        files: List of files to upload
        metadatas_json: JSON string containing metadata for each file
        chunk_size: Maximum number of characters in each chunk (default: 3000, range: 10-10000)
        chunk_overlap: Number of overlapping characters between chunks (default: 200, range: 0-5000)
        rebuild_wiki: Whether to rebuild LLM Wiki immediately after upload.
    """
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=400,
            detail="chunk_overlap must be less than chunk_size.",
        )

    # Validate file sizes
    for file in files:
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File '{file.filename}' exceeds maximum size of 50MB.",
            )
        await file.seek(0)  # Reset for later reading

    # If no metadata JSON is provided, fill with None
    if not metadatas_json:
        metadatas: list[dict] | list[None] = [None] * len(files)
    else:
        try:
            # This will both parse the JSON and check the Python types
            # (i.e. that it's a list, and every item is a dict)
            metadatas = _metadata_adapter.validate_json(metadatas_json)
        except ValidationError as e:
            # Pydantic errors include exactly what went wrong
            raise HTTPException(status_code=400, detail=e.errors())
        # Now just check that the list length matches
        if len(metadatas) != len(files):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Number of metadata objects ({len(metadatas)}) "
                    f"does not match number of files ({len(files)})."
                ),
            )

    docs_to_index: list[Document] = []
    processed_files_count = 0
    failed_files = []
    paper_card_warnings: list[str] = []

    # Pair files with their corresponding metadata
    for file, metadata in zip(files, metadatas, strict=False):
        try:
            # Pass metadata and chunk parameters to process_document
            langchain_docs = await process_document(
                file,
                metadata=metadata,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                collection_id=str(collection_id),
                paper_card_warnings=paper_card_warnings,
            )
            if langchain_docs:
                docs_to_index.extend(langchain_docs)
                processed_files_count += 1
            else:
                logger.info(
                    f"Warning: File {file.filename} resulted "
                    f"in no processable documents."
                )
                # Decide if this constitutes a failure
                # failed_files.append(file.filename)

        except Exception as proc_exc:
            # Log the error and the file that caused it
            logger.info(f"Error processing file {file.filename}: {proc_exc}")
            failed_files.append(file.filename)
            # Decide on behavior: continue processing others or fail fast?
            # For now, let's collect failures and report them, but continue processing.

    # If after processing all files, none yielded documents, raise error
    if not docs_to_index:
        error_detail = "Failed to process any documents from the provided files."
        if failed_files:
            error_detail += f" Files that failed processing: {', '.join(failed_files)}."
        raise HTTPException(status_code=400, detail=error_detail)

    # If some files failed but others succeeded, proceed with adding successful ones
    # but maybe inform the user about the failures.
    try:
        collection = Collection(collection_id=str(collection_id))
        added_ids = await collection.upsert(docs_to_index)
        if not added_ids:
            # This might indicate a problem with the vector store itself
            raise HTTPException(
                status_code=500,
                detail="Failed to add document(s) to vector store after processing.",
            )

        # Construct response message
        success_message = (
            f"{len(added_ids)} document chunk(s) from "
            f"{processed_files_count} file(s) added successfully."
        )
        response_data = {
            "success": True,
            "message": success_message,
            "added_chunk_ids": added_ids,
        }

        if rebuild_wiki:
            try:
                wiki_result = await rebuild_llm_wiki(str(collection_id))
            except Exception as wiki_exc:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "success": False,
                        "error": "documents_indexed_wiki_rebuild_failed",
                        "message": "Documents were indexed, but LLM Wiki rebuild failed.",
                        "documents_indexed": True,
                        "added_chunk_ids": added_ids,
                        "wiki_rebuild_error": str(wiki_exc),
                    },
                )

            response_data["llm_wiki"] = wiki_result.model_dump(mode="json")
        else:
            response_data["llm_wiki"] = {
                "skipped": True,
                "recovery": "Call rebuild_llm_wiki(collection_id) after all uploads finish.",
            }

        warnings: list[str] = []
        if failed_files:
            warnings.append(f"Processing failed for files: {', '.join(failed_files)}")
        warnings.extend(paper_card_warnings)
        if warnings:
            response_data["warnings"] = " | ".join(warnings)
            # Consider if partial success should change the overall status/message

        return response_data

    except HTTPException as http_exc:
        # Reraise HTTPExceptions from add_documents_to_vectorstore or previous checks
        raise http_exc
    except Exception as add_exc:
        # Handle exceptions during the vector store addition process
        logger.info(f"Error adding documents to vector store: {add_exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add documents to vector store: {add_exc!s}",
        )


@router.get(
    "/collections/{collection_id}/documents", response_model=list[DocumentResponse]
)
async def documents_list(
    collection_id: UUID,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """Lists documents within a specific collection."""
    collection = Collection(collection_id=str(collection_id))
    return await collection.list(limit=limit, offset=offset)


@router.get(
    "/collections/{collection_id}/documents/{document_id}",
    response_model=DocumentResponse,
)
async def documents_get(
    collection_id: UUID,
    document_id: str,
    file_id: Annotated[
        str | None,
        Query(description="Optional file_id from an LLM Wiki source_ref."),
    ] = None,
):
    """Fetch one document chunk by ID, optionally validating its source file."""
    collection = Collection(collection_id=str(collection_id))
    document = await collection.get(document_id)
    if file_id is not None and (document.get("metadata") or {}).get("file_id") != file_id:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.delete(
    "/collections/{collection_id}/documents/{document_id}",
    response_model=dict[str, bool],
)
async def documents_delete(
    collection_id: UUID,
    document_id: str,
    delete_by: Annotated[
        str,
        Query(description="Delete by 'document_id' or 'file_id'"),
    ] = "document_id",
):
    """Deletes a specific document from a collection by its ID.

    Args:
        collection_id: The collection UUID containing the document(s)
        document_id: The ID to delete by (either document ID or file ID)
        delete_by: Specifies whether to delete by 'document_id' (single chunk) or 'file_id' (all chunks from file)
    """
    if delete_by not in {"document_id", "file_id"}:
        raise HTTPException(
            status_code=400,
            detail="delete_by must be either 'document_id' or 'file_id'.",
        )

    collection = Collection(collection_id=str(collection_id))
    if delete_by == "file_id":
        deleted_count = await collection.delete(file_id=document_id)
    else:  # Default to document_id
        deleted_count = await collection.delete(document_id=document_id)

    if deleted_count > 0:
        await _rebuild_llm_wiki_after_delete_or_raise(
            collection_id,
            deleted_count=deleted_count,
        )

    return {"success": True}


@router.post(
    "/collections/{collection_id}/documents/search", response_model=list[SearchResult]
)
async def documents_search(
    collection_id: UUID,
    search_query: SearchQuery,
):
    """Search for documents within a specific collection."""
    if not search_query.query:
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    collection = Collection(collection_id=str(collection_id))

    results = await collection.search(
        search_query.query,
        limit=search_query.limit or 10,
        search_type=search_query.search_type,
        filter=search_query.filter,
        min_score=search_query.min_score,
    )
    return results
