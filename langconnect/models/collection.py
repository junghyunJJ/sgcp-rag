from typing import Any

from pydantic import BaseModel, Field

# =====================
# Collection Schemas
# =====================


class CollectionCreate(BaseModel):
    """Schema for creating a new collection."""

    name: str = Field(..., description="The unique name of the collection.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata for the collection."
    )


class CollectionUpdate(BaseModel):
    """Schema for updating an existing collection."""

    name: str | None = Field(None, description="New name for the collection.")
    metadata: dict[str, Any] | None = Field(
        None, description="Updated metadata for the collection."
    )


class CollectionResponse(BaseModel):
    """Schema for representing a collection from PGVector."""

    # PGVector table has uuid (id), name (str), and cmetadata (JSONB)
    # We get these from list/get db functions
    uuid: str = Field(
        ..., description="The unique identifier of the collection in PGVector."
    )
    name: str = Field(..., description="The name of the collection.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata associated with the collection."
    )
    document_count: int = Field(0, description="The number of documents in the collection.")
    chunk_count: int = Field(0, description="The number of chunks in the collection.")

    class Config:
        from_attributes = True
