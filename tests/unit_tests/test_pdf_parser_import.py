"""Test PyMuPDF4LLM parser import and configuration."""


def test_pymupdf4llm_parser_configured() -> None:
    """Test that PyMuPDF4LLMParser is correctly configured in document_processor."""
    from langconnect.parsers.pymupdf_parser import PyMuPDF4LLMParser
    from langconnect.services.document_processor import HANDLERS

    # Verify PyMuPDF4LLMParser is used for PDFs
    assert "application/pdf" in HANDLERS
    assert isinstance(HANDLERS["application/pdf"], PyMuPDF4LLMParser)


def test_pymupdf4llm_import() -> None:
    """Test that we can import PyMuPDF4LLMParser."""
    from langconnect.parsers.pymupdf_parser import PyMuPDF4LLMParser

    # Verify the parser can be instantiated
    parser = PyMuPDF4LLMParser()
    assert parser is not None
