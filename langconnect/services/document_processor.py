import logging
import uuid

from fastapi import UploadFile
from langchain_community.document_loaders.parsers import (
    BS4HTMLParser,
)
from langchain_community.document_loaders.parsers.generic import MimeTypeBasedParser
from langchain_community.document_loaders.parsers.msword import MsWordParser
from langchain_community.document_loaders.parsers.txt import TextParser
from langconnect.parsers.pymupdf_parser import PyMuPDF4LLMParser
from langchain_core.documents.base import Blob, Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

LOGGER = logging.getLogger(__name__)

# Document Parser Configuration
HANDLERS = {
    "application/pdf": PyMuPDF4LLMParser(),
    "text/plain": TextParser(),
    "text/html": BS4HTMLParser(),
    "text/markdown": TextParser(),  # Markdown files
    "text/x-markdown": TextParser(),  # Alternative markdown MIME type
    "application/msword": MsWordParser(),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
        MsWordParser()
    ),
}

SUPPORTED_MIMETYPES = sorted(HANDLERS.keys())

MIMETYPE_BASED_PARSER = MimeTypeBasedParser(
    handlers=HANDLERS,
    fallback_parser=None,
)


async def process_document(
    file: UploadFile,
    metadata: dict | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Process an uploaded file into LangChain documents."""
    # Generate a unique ID for this file processing instance
    file_id = uuid.uuid4()

    contents = await file.read()

    # Determine the actual mime type
    mime_type = file.content_type or "text/plain"

    # Handle application/octet-stream by checking file extension
    if mime_type == "application/octet-stream" and file.filename:
        filename_lower = file.filename.lower()
        if filename_lower.endswith(".md") or filename_lower.endswith(".markdown"):
            mime_type = "text/markdown"
        elif filename_lower.endswith(".txt"):
            mime_type = "text/plain"
        elif filename_lower.endswith(".html") or filename_lower.endswith(".htm"):
            mime_type = "text/html"
        elif filename_lower.endswith(".pdf"):
            mime_type = "application/pdf"
        elif filename_lower.endswith(".doc"):
            mime_type = "application/msword"
        elif filename_lower.endswith(".docx"):
            mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    blob = Blob(data=contents, mimetype=mime_type)

    docs = MIMETYPE_BASED_PARSER.parse(blob)

    # Add provided metadata to each document
    if metadata:
        for doc in docs:
            # Ensure metadata attribute exists and is a dict
            if not hasattr(doc, "metadata") or not isinstance(doc.metadata, dict):
                doc.metadata = {}
            # Update with provided metadata, preserving existing keys if not overridden
            doc.metadata.update(metadata)

    # Create text splitter with provided parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Split documents
    split_docs = text_splitter.split_documents(docs)

    # Add the generated file_id to all split documents' metadata
    for split_doc in split_docs:
        if not hasattr(split_doc, "metadata") or not isinstance(
            split_doc.metadata, dict
        ):
            split_doc.metadata = {}  # Initialize if it doesn't exist
        split_doc.metadata["file_id"] = str(
            file_id
        )  # Store as string for compatibility

    return split_docs
