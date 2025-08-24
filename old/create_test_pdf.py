#!/usr/bin/env python3
"""Create a test PDF for pymupdf4llm testing."""

try:
    from fpdf import FPDF
    
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(200, 10, txt="Test PDF Document", ln=1, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt="This is a test document to verify pymupdf4llm works correctly.", ln=1)
    
    pdf.ln(5)
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(200, 10, txt="Features", ln=1)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="- Converts PDFs to markdown format", ln=1)
    pdf.cell(200, 10, txt="- Preserves document structure", ln=1)
    pdf.cell(200, 10, txt="- Suitable for LLM processing", ln=1)
    
    pdf.ln(5)
    pdf.cell(200, 10, txt="This confirms our PDF to markdown conversion is working.", ln=1)
    
    pdf.output("test_pymupdf_proper.pdf")
    print("PDF created successfully as test_pymupdf_proper.pdf")
    
except ImportError:
    # Alternative using HTML to PDF conversion
    import os
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head><title>Test PDF</title></head>
    <body>
        <h1>Test PDF Document</h1>
        <p>This is a test document to verify pymupdf4llm works correctly.</p>
        
        <h2>Features</h2>
        <ul>
            <li>Converts PDFs to markdown format</li>
            <li>Preserves document structure</li>
            <li>Suitable for LLM processing</li>
        </ul>
        
        <p>This confirms our PDF to markdown conversion is working.</p>
    </body>
    </html>
    """
    
    # Write HTML file
    with open("test.html", "w") as f:
        f.write(html_content)
    
    # Try to convert with wkhtmltopdf if available
    if os.system("which wkhtmltopdf > /dev/null 2>&1") == 0:
        os.system("wkhtmltopdf test.html test_pymupdf_proper.pdf")
        os.remove("test.html")
        print("PDF created with wkhtmltopdf")
    else:
        print("Neither fpdf nor wkhtmltopdf available. Using text file instead.")
        # Just create a text file for testing
        with open("test_pymupdf_proper.txt", "w") as f:
            f.write("""# Test PDF Document

This is a test document to verify pymupdf4llm works correctly.

## Features

- Converts PDFs to markdown format
- Preserves document structure  
- Suitable for LLM processing

This confirms our PDF to markdown conversion is working.""")