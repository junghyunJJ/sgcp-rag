"""PyMuPDF4LLM-based PDF parser that converts PDFs to markdown."""

import logging
from collections.abc import Iterator

try:
    import pymupdf
    import pymupdf4llm
except ImportError as e:
    raise ImportError(
        "pymupdf4llm is required for PDF parsing. "
        "Please install it with: pip install pymupdf4llm"
    ) from e

from langchain_core.document_loaders.base import BaseBlobParser
from langchain_core.documents import Document
from langchain_core.documents.base import Blob

from langconnect.parsers.pdf_markdown_cleanup import (
    PDF_MARKDOWN_CLEANUP_VERSION,
    clean_pdf_markdown,
)

logger = logging.getLogger(__name__)


class PyMuPDF4LLMParser(BaseBlobParser):
    """Parser that uses pymupdf4llm to convert PDFs to markdown format.

    This parser converts PDF documents to markdown, which preserves
    formatting like headers, lists, tables, etc. This is especially
    useful for LLM processing as markdown is a structured format.
    """

    def __init__(self, *, clean_markdown: bool = True, **kwargs: object) -> None:
        """Initialize the parser.

        Args:
            clean_markdown: Remove conservative PDF boilerplate from markdown.
            **kwargs: Additional keyword arguments to pass to pymupdf4llm.to_markdown()
        """
        self.clean_markdown = clean_markdown
        self.kwargs = kwargs

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob into Documents.

        Args:
            blob: The blob to parse

        Yields:
            Document: Parsed documents with markdown content
        """
        try:
            # Get the PDF data
            pdf_data = blob.data if isinstance(blob.data, bytes) else blob.data.encode()

            # Create a PyMuPDF document from bytes
            doc = pymupdf.open(stream=pdf_data, filetype="pdf")

            # Convert PDF to markdown using the document object
            try:
                markdown_text = pymupdf4llm.to_markdown(doc, **self.kwargs)
            finally:
                doc.close()
            if self.clean_markdown:
                markdown_text = clean_pdf_markdown(markdown_text)

            # Create metadata
            metadata = {
                "source": blob.source or "pymupdf4llm",
                "format": "markdown",
                "parser": "PyMuPDF4LLMParser",
            }
            if self.clean_markdown:
                metadata["markdown_cleanup"] = PDF_MARKDOWN_CLEANUP_VERSION

            # If blob has existing metadata, merge it
            if blob.metadata:
                metadata.update(blob.metadata)

            # Yield a single document with the entire markdown content
            # The text splitter will handle chunking later
            yield Document(
                page_content=markdown_text,
                metadata=metadata,
            )

        except Exception:
            logger.exception("Error parsing PDF with pymupdf4llm")
            # Do not yield an empty document — let the caller handle the absence
            return
