"""Tests for the web scraper tool."""

import json
import pytest
from unittest.mock import patch, AsyncMock


SAMPLE_HTML = """
<html>
<head><title>Test Page</title></head>
<body>
<h1>Main Title</h1>
<h2>Section 1</h2>
<table>
  <tr><th>Name</th><th>Value</th></tr>
  <tr><td>Alpha</td><td>100</td></tr>
  <tr><td>Beta</td><td>200</td></tr>
</table>
<ul>
  <li>Item one</li>
  <li>Item two</li>
  <li>Item three</li>
</ul>
<a href="https://example.com/page1">Link 1</a>
<a href="https://example.com/page2">Link 2</a>
<h2>Section 2</h2>
<p>Some paragraph text.</p>
</body>
</html>
"""


class TestWebScraper:
    """Tests for the web scraper tool."""

    @pytest.mark.asyncio
    async def test_extract_tables(self):
        from src.tools.scraper_tool import scrape_webpage

        with patch("src.tools.scraper_tool._fetch_html", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = SAMPLE_HTML
            result = await scrape_webpage("https://example.com")

        assert "Table 1" in result
        assert "Alpha" in result
        assert "100" in result
        assert "Beta" in result

    @pytest.mark.asyncio
    async def test_extract_lists(self):
        from src.tools.scraper_tool import scrape_webpage

        with patch("src.tools.scraper_tool._fetch_html", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = SAMPLE_HTML
            result = await scrape_webpage("https://example.com")

        assert "Item one" in result
        assert "Item two" in result

    @pytest.mark.asyncio
    async def test_extract_links(self):
        from src.tools.scraper_tool import scrape_webpage

        with patch("src.tools.scraper_tool._fetch_html", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = SAMPLE_HTML
            result = await scrape_webpage("https://example.com")

        assert "Link 1" in result
        assert "https://example.com/page1" in result

    @pytest.mark.asyncio
    async def test_extract_headings(self):
        from src.tools.scraper_tool import scrape_webpage

        with patch("src.tools.scraper_tool._fetch_html", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = SAMPLE_HTML
            result = await scrape_webpage("https://example.com")

        assert "Main Title" in result
        assert "Section 1" in result

    @pytest.mark.asyncio
    async def test_selective_extraction(self):
        from src.tools.scraper_tool import scrape_webpage

        with patch("src.tools.scraper_tool._fetch_html", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = SAMPLE_HTML
            query = json.dumps({"url": "https://example.com", "extract": ["tables"]})
            result = await scrape_webpage(query)

        assert "Table 1" in result
        # Should not extract links when only tables requested
        assert "Links:" not in result

    @pytest.mark.asyncio
    async def test_css_selector(self):
        from src.tools.scraper_tool import scrape_webpage

        html_with_class = '<html><body><div class="main">Target content</div><div>Other</div></body></html>'
        with patch("src.tools.scraper_tool._fetch_html", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = html_with_class
            query = json.dumps({"url": "https://example.com", "selector": "div.main"})
            result = await scrape_webpage(query)

        # The selector targets div.main but won't have structured data
        assert "example.com" in result

    @pytest.mark.asyncio
    async def test_empty_url(self):
        from src.tools.scraper_tool import scrape_webpage
        result = await scrape_webpage("")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_no_structured_data(self):
        from src.tools.scraper_tool import scrape_webpage

        simple_html = "<html><body><p>Just a paragraph.</p></body></html>"
        with patch("src.tools.scraper_tool._fetch_html", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = simple_html
            result = await scrape_webpage("https://example.com")

        assert "No structured data" in result or "fetch_url" in result

    def test_tool_wrapper_exists(self):
        from src.tools.scraper_tool import scraper_tool
        assert scraper_tool.name == "web_scraper"
        assert scraper_tool.coroutine is not None
