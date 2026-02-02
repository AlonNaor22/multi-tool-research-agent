"""PDF reader tool for the research agent.

Fetches and extracts text content from PDF files at URLs.

Useful for reading research papers, reports, documentation, etc.
"""

import io
import re
import requests
from typing import Optional
from langchain_core.tools import Tool

from src.utils import retry_on_error

# Try to import PyPDF2 or pypdf
try:
    from pypdf import PdfReader
    PDF_LIBRARY = "pypdf"
except ImportError:
    try:
        from PyPDF2 import PdfReader
        PDF_LIBRARY = "PyPDF2"
    except ImportError:
        PdfReader = None
        PDF_LIBRARY = None


@retry_on_error(max_retries=2, delay=1.0)
def fetch_pdf(url: str, timeout: int = 30) -> bytes:
    """
    Fetch PDF content from a URL.

    Args:
        url: URL to the PDF file
        timeout: Request timeout in seconds

    Returns:
        PDF file content as bytes
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/pdf,*/*",
    }

    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    # Verify it's actually a PDF
    content_type = response.headers.get("Content-Type", "")
    if "pdf" not in content_type.lower() and not response.content[:4] == b'%PDF':
        raise ValueError(f"URL does not point to a PDF file (Content-Type: {content_type})")

    return response.content


def extract_text_from_pdf(pdf_content: bytes, max_pages: Optional[int] = None) -> str:
    """
    Extract text from PDF content.

    Args:
        pdf_content: PDF file content as bytes
        max_pages: Maximum number of pages to extract (None = all)

    Returns:
        Extracted text content
    """
    if PdfReader is None:
        return "Error: PDF reading library not installed. Please install pypdf: pip install pypdf"

    try:
        pdf_file = io.BytesIO(pdf_content)
        reader = PdfReader(pdf_file)

        total_pages = len(reader.pages)
        pages_to_read = min(total_pages, max_pages) if max_pages else total_pages

        text_parts = []
        text_parts.append(f"[PDF Document - {total_pages} pages]")

        # Extract metadata if available
        metadata = reader.metadata
        if metadata:
            if metadata.title:
                text_parts.append(f"Title: {metadata.title}")
            if metadata.author:
                text_parts.append(f"Author: {metadata.author}")
            if metadata.subject:
                text_parts.append(f"Subject: {metadata.subject}")
            text_parts.append("")

        # Extract text from pages
        for i in range(pages_to_read):
            page = reader.pages[i]
            page_text = page.extract_text()

            if page_text:
                text_parts.append(f"--- Page {i + 1} ---")
                text_parts.append(page_text.strip())
                text_parts.append("")

        if max_pages and total_pages > max_pages:
            text_parts.append(f"[... {total_pages - max_pages} more pages not shown ...]")

        return "\n".join(text_parts)

    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"


def clean_text(text: str, max_length: int = 15000) -> str:
    """
    Clean and truncate extracted text.

    Args:
        text: Raw extracted text
        max_length: Maximum length to return

    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # Remove common PDF artifacts
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

    # Truncate if needed
    if len(text) > max_length:
        text = text[:max_length] + f"\n\n[... Content truncated at {max_length} characters ...]"

    return text.strip()


def read_pdf(input_str: str) -> str:
    """
    Fetch and read a PDF from a URL.

    Supports formats:
    - "https://example.com/paper.pdf"
    - "url: https://example.com/paper.pdf"
    - "5 pages: https://example.com/paper.pdf" (limit pages)
    - "summary: https://example.com/paper.pdf" (first 3 pages only)

    Args:
        input_str: URL to the PDF, optionally with options

    Returns:
        Extracted text content or error message
    """
    input_str = input_str.strip()

    if not input_str:
        return "Error: No URL provided"

    # Command: help
    if input_str.lower() in ("help", "?"):
        return _get_help()

    # Parse options
    max_pages = None
    url = input_str

    # Check for "N pages:" prefix
    pages_match = re.match(r'(\d+)\s+pages?:\s*(.+)', input_str, re.IGNORECASE)
    if pages_match:
        max_pages = int(pages_match.group(1))
        url = pages_match.group(2)

    # Check for "summary:" prefix (first 3 pages)
    if url.lower().startswith("summary:"):
        max_pages = 3
        url = url[8:].strip()

    # Check for "url:" prefix
    if url.lower().startswith("url:"):
        url = url[4:].strip()

    # Validate URL
    if not url.startswith(("http://", "https://")):
        return f"Error: Invalid URL '{url}'. Must start with http:// or https://"

    try:
        # Fetch the PDF
        pdf_content = fetch_pdf(url)

        # Extract text
        text = extract_text_from_pdf(pdf_content, max_pages)

        # Clean and return
        return clean_text(text)

    except requests.exceptions.RequestException as e:
        return f"Error fetching PDF: {str(e)}"
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


def _get_help() -> str:
    """Return help text."""
    library_status = f"PDF library: {PDF_LIBRARY}" if PDF_LIBRARY else "PDF library: NOT INSTALLED (pip install pypdf)"

    return f"""PDF Reader Help:

FORMAT:
  https://example.com/document.pdf
  5 pages: https://arxiv.org/pdf/1234.5678.pdf
  summary: https://example.com/report.pdf (first 3 pages)

OPTIONS:
  N pages: url  - Only read first N pages
  summary: url  - Quick summary (first 3 pages only)
  url: url      - Explicit URL prefix

RETURNS:
  - PDF metadata (title, author, subject)
  - Full text content from all pages
  - Page separators for reference

TIPS:
  - Use with arxiv_search to read paper abstracts/content
  - Use "summary:" for quick overview of long documents
  - Large PDFs are automatically truncated to ~15,000 characters

{library_status}"""


# Create the LangChain Tool wrapper
pdf_tool = Tool(
    name="pdf_reader",
    func=read_pdf,
    description=(
        "Read and extract text content from PDF files at URLs. "
        "\n\nFORMAT: 'https://example.com/paper.pdf', '5 pages: URL', 'summary: URL'"
        "\n\nRETURNS: PDF metadata (title, author) and full text content."
        "\n\nUSE FOR: Reading research papers, reports, documentation from arxiv, journals, etc."
        "\n\nTIP: Use 'summary:' prefix for quick overview of long documents."
    )
)
