"""Tests for src/tools/pdf_tool.py -- PDF document reading."""

import io
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tests.conftest import AsyncMockResponse


class TestPdfReader:
    """Test PDF reading with mocked HTTP and PDF library."""

    async def test_missing_pdf_library(self):
        with patch("src.tools.pdf_tool.PDF_LIBRARY", None):
            from src.tools.pdf_tool import extract_text_from_pdf
            result = extract_text_from_pdf(b"fake pdf content")
            assert "not installed" in result.lower() or "install" in result.lower()

    async def test_invalid_url(self):
        from src.tools.pdf_tool import read_pdf
        result = await read_pdf("")
        assert len(result) > 0  # Should return error, not crash

    async def test_successful_pdf_read_pdfplumber(self):
        """Test PDF reading with pdfplumber mocked."""
        pdf_bytes = b"%PDF-1.4 fake pdf content"
        mock_resp = AsyncMockResponse(
            content=pdf_bytes,
            status=200,
            headers={"Content-Type": "application/pdf"}
        )
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        # Mock pdfplumber.open context manager
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is the extracted PDF text."

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {"Title": "Test PDF", "Author": "Test Author"}
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("src.tools.pdf_tool.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session), \
             patch("src.tools.pdf_tool.pdfplumber") as mock_pdfplumber:
            mock_pdfplumber.open.return_value = mock_pdf
            from src.tools.pdf_tool import read_pdf
            result = await read_pdf("https://example.com/paper.pdf")

            assert "extracted PDF text" in result or "Test PDF" in result

    async def test_handles_download_error(self):
        import aiohttp
        mock_session = MagicMock()
        mock_session.get.side_effect = aiohttp.ClientError("failed")

        with patch("src.tools.pdf_tool.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.pdf_tool import read_pdf
            result = await read_pdf("https://example.com/paper.pdf")

            assert "Error" in result or "error" in result.lower()

    def test_extract_text_pdfplumber(self):
        """Test extract_text_from_pdf with pdfplumber."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page content here."

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {"Title": "My Paper", "Author": "Author Name"}
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("src.tools.pdf_tool.pdfplumber") as mock_pdfplumber, \
             patch("src.tools.pdf_tool.PDF_LIBRARY", "pdfplumber"):
            mock_pdfplumber.open.return_value = mock_pdf
            from src.tools.pdf_tool import _extract_with_pdfplumber
            result = _extract_with_pdfplumber(b"fake pdf bytes")

            assert "Page content" in result or "My Paper" in result

    def test_max_pages_limit(self):
        """Test that max_pages limits extraction."""
        mock_pages = [MagicMock() for _ in range(5)]
        for i, page in enumerate(mock_pages):
            page.extract_text.return_value = f"Page {i + 1} content."

        mock_pdf = MagicMock()
        mock_pdf.pages = mock_pages
        mock_pdf.metadata = {}
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("src.tools.pdf_tool.pdfplumber") as mock_pdfplumber, \
             patch("src.tools.pdf_tool.PDF_LIBRARY", "pdfplumber"):
            mock_pdfplumber.open.return_value = mock_pdf
            from src.tools.pdf_tool import _extract_with_pdfplumber
            result = _extract_with_pdfplumber(b"fake", max_pages=2)

            assert "Page 1 content" in result
            assert "Page 2 content" in result
            assert "Page 3 content" not in result
            assert "3 more pages not shown" in result

    def test_clean_text_truncation(self):
        from src.tools.pdf_tool import clean_text
        long_text = "a" * 20000
        result = clean_text(long_text, max_length=100)
        assert len(result) < 200
        assert "truncated" in result.lower()

    async def test_help_command(self):
        from src.tools.pdf_tool import read_pdf
        result = await read_pdf("help")
        assert "FORMAT" in result

    async def test_summary_prefix(self):
        """Test that summary: prefix sets max_pages to 3."""
        pdf_bytes = b"%PDF-1.4 fake"
        mock_resp = AsyncMockResponse(
            content=pdf_bytes,
            status=200,
            headers={"Content-Type": "application/pdf"}
        )
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page content."

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page] * 10
        mock_pdf.metadata = {}
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("src.tools.pdf_tool.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session), \
             patch("src.tools.pdf_tool.pdfplumber") as mock_pdfplumber:
            mock_pdfplumber.open.return_value = mock_pdf
            from src.tools.pdf_tool import read_pdf
            result = await read_pdf("summary: https://example.com/paper.pdf")

            assert "more pages not shown" in result
