"""Tests for src/tools/pdf_tool.py — PDF document reading."""

import io
import pytest
from unittest.mock import patch, MagicMock
from tests.conftest import MockResponse


class TestPdfReader:
    """Test PDF reading with mocked HTTP and PDF library."""

    def test_missing_pdf_library(self):
        with patch("src.tools.pdf_tool.PdfReader", None):
            from src.tools.pdf_tool import extract_text_from_pdf
            result = extract_text_from_pdf(b"fake pdf content")
            assert "not installed" in result.lower() or "install" in result.lower()

    def test_invalid_url(self):
        from src.tools.pdf_tool import read_pdf
        result = read_pdf("")
        assert len(result) > 0  # Should return error, not crash

    def test_successful_pdf_read(self):
        # Mock the HTTP response
        pdf_bytes = b"%PDF-1.4 fake pdf content"
        mock_resp = MockResponse(
            content=pdf_bytes,
            status_code=200,
            headers={"Content-Type": "application/pdf"}
        )

        # Mock PdfReader with proper metadata structure
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is the extracted PDF text."

        mock_metadata = MagicMock()
        mock_metadata.title = "Test PDF"
        mock_metadata.author = "Test Author"
        mock_metadata.subject = None

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader.metadata = mock_metadata

        with patch("src.tools.pdf_tool.requests.get", return_value=mock_resp), \
             patch("src.tools.pdf_tool.PdfReader", return_value=mock_reader):
            from src.tools.pdf_tool import read_pdf
            result = read_pdf("https://example.com/paper.pdf")

            assert "extracted PDF text" in result or "Test PDF" in result

    def test_handles_download_error(self):
        import requests
        with patch("src.tools.pdf_tool.requests.get",
                   side_effect=requests.exceptions.ConnectionError("failed")):
            from src.tools.pdf_tool import read_pdf
            result = read_pdf("https://example.com/paper.pdf")

            assert "Error" in result or "error" in result.lower()

    def test_extract_text_from_valid_pdf(self):
        """Test the extract_text_from_pdf function directly."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page content here."

        mock_metadata = MagicMock()
        mock_metadata.title = "My Paper"
        mock_metadata.author = "Author Name"
        mock_metadata.subject = None

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader.metadata = mock_metadata

        with patch("src.tools.pdf_tool.PdfReader", return_value=mock_reader):
            from src.tools.pdf_tool import extract_text_from_pdf
            result = extract_text_from_pdf(b"fake pdf bytes")

            assert "Page content" in result or "My Paper" in result
