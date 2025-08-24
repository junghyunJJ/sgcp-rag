"""Test PDFPlumberParser import and configuration."""


def test_pdf_plumber_parser_configured() -> None:
    """Test that PDFPlumberParser is correctly configured in document_processor."""
    from langchain_community.document_loaders.parsers import PDFPlumberParser

    from langconnect.services.document_processor import HANDLERS

    # Verify PDFPlumberParser is used for PDFs
    assert "application/pdf" in HANDLERS
    assert isinstance(HANDLERS["application/pdf"], PDFPlumberParser)


def test_pdf_plumber_import() -> None:
    """Test that we can import PDFPlumberParser."""
    from langchain_community.document_loaders.parsers import PDFPlumberParser

    # Verify the parser can be instantiated
    parser = PDFPlumberParser()
    assert parser is not None
