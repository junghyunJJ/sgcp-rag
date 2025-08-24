"""PyMuPDF4LLM-based PDF parser that converts PDFs to markdown."""

import logging
from typing import Iterator

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

logger = logging.getLogger(__name__)


class PyMuPDF4LLMParser(BaseBlobParser):
    """Parser that uses pymupdf4llm to convert PDFs to markdown format.
    
    This parser converts PDF documents to markdown, which preserves
    formatting like headers, lists, tables, etc. This is especially
    useful for LLM processing as markdown is a structured format.
    """

    def __init__(self, **kwargs):
        """Initialize the parser.
        
        Args:
            **kwargs: Additional keyword arguments to pass to pymupdf4llm.to_markdown()
        """
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
            if isinstance(blob.data, bytes):
                pdf_data = blob.data
            else:
                # If it's a string, encode it
                pdf_data = blob.data.encode()
            
            # Create a PyMuPDF document from bytes
            doc = pymupdf.open(stream=pdf_data, filetype="pdf")
            
            # Convert PDF to markdown using the document object
            markdown_text = pymupdf4llm.to_markdown(doc, **self.kwargs)
            
            # Close the document
            doc.close()
            
            # Create metadata
            metadata = {
                "source": blob.source or "pymupdf4llm",
                "format": "markdown",
                "parser": "PyMuPDF4LLMParser",
            }
            
            # If blob has existing metadata, merge it
            if blob.metadata:
                metadata.update(blob.metadata)
            
            # Yield a single document with the entire markdown content
            # The text splitter will handle chunking later
            yield Document(
                page_content=markdown_text,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error parsing PDF with pymupdf4llm: {e}")
            # Yield an empty document on error to maintain consistency
            yield Document(
                page_content="",
                metadata={
                    "source": blob.source or "pymupdf4llm",
                    "error": str(e),
                    "parser": "PyMuPDF4LLMParser",
                }
            )