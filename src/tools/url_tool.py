"""URL content fetcher tool for the research agent.

Fetches and extracts readable text content from web pages and PDFs.
Uses aiohttp for HTTP, BeautifulSoup for HTML parsing, and pypdf for PDFs.
No API key required.

Features:
- HTML page content extraction
- PDF document text extraction
- Metadata extraction (author, date, description)
- Smart content detection (main article vs full page)
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
from io import BytesIO
from langchain_core.tools import Tool
from src.utils import async_retry_on_error, get_aiohttp_session, make_sync
from src.constants import DEFAULT_USER_AGENT, DEFAULT_HTTP_TIMEOUT

# Try to import pypdf for PDF support
try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Content limits
MAX_CONTENT_CHARS = 5000
MAX_PDF_PAGES = 20


def _extract_metadata(soup: BeautifulSoup) -> dict:
    """Extract metadata from HTML page."""
    metadata = {}

    # Try to get author
    author_meta = (
        soup.find("meta", {"name": "author"}) or
        soup.find("meta", {"property": "article:author"}) or
        soup.find("meta", {"name": "twitter:creator"})
    )
    if author_meta and author_meta.get("content"):
        metadata["author"] = author_meta["content"]

    # Try to get publish date
    date_meta = (
        soup.find("meta", {"property": "article:published_time"}) or
        soup.find("meta", {"name": "date"}) or
        soup.find("meta", {"name": "publish_date"}) or
        soup.find("time", {"datetime": True})
    )
    if date_meta:
        if date_meta.name == "time":
            metadata["date"] = date_meta.get("datetime", "")[:10]
        elif date_meta.get("content"):
            metadata["date"] = date_meta["content"][:10]

    # Try to get description
    desc_meta = (
        soup.find("meta", {"name": "description"}) or
        soup.find("meta", {"property": "og:description"})
    )
    if desc_meta and desc_meta.get("content"):
        metadata["description"] = desc_meta["content"][:200]

    return metadata


def _extract_pdf_content(content: bytes, url: str) -> str:
    """Extract text content from a PDF file."""
    if not PDF_SUPPORT:
        return (
            "PDF detected but pypdf is not installed. "
            "Install it with: pip install pypdf"
        )

    try:
        pdf_file = BytesIO(content)
        reader = PdfReader(pdf_file)

        # Get PDF metadata
        metadata_parts = []
        if reader.metadata:
            if reader.metadata.title:
                metadata_parts.append(f"**Title:** {reader.metadata.title}")
            if reader.metadata.author:
                metadata_parts.append(f"**Author:** {reader.metadata.author}")

        metadata_parts.append(f"**Pages:** {len(reader.pages)}")

        # Extract text from pages (limit to avoid huge outputs)
        pages_to_read = min(len(reader.pages), MAX_PDF_PAGES)
        text_parts = []

        for i in range(pages_to_read):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text_parts.append(f"--- Page {i + 1} ---\n{page_text.strip()}")

        if len(reader.pages) > MAX_PDF_PAGES:
            text_parts.append(f"\n[Showing first {MAX_PDF_PAGES} of {len(reader.pages)} pages]")

        # Combine metadata and content
        content_text = "\n\n".join(text_parts)

        # Truncate if too long
        if len(content_text) > MAX_CONTENT_CHARS:
            content_text = content_text[:MAX_CONTENT_CHARS] + "\n\n[Content truncated]"

        return "\n".join(metadata_parts) + "\n\n**Content:**\n" + content_text

    except Exception as e:
        return f"Error extracting PDF content: {str(e)}"


def _extract_html_content(html: str, url: str) -> str:
    """Extract readable content from HTML."""
    soup = BeautifulSoup(html, 'html.parser')

    # Remove script, style, and navigation elements
    for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript']):
        element.decompose()

    # Get page title
    title = soup.find('title')
    title_text = title.get_text(strip=True) if title else "No title"

    # Get metadata
    metadata = _extract_metadata(soup)

    # Try to find main content areas
    main_content = (
        soup.find('main') or
        soup.find('article') or
        soup.find('div', class_='content') or
        soup.find('div', class_='post') or
        soup.find('div', class_='article') or
        soup.find('div', id='content')
    )

    if main_content:
        content = main_content.get_text(separator='\n', strip=True)
    else:
        # Fallback: get all paragraph text
        paragraphs = soup.find_all('p')
        content = '\n'.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

    # If still no content, get body text
    if not content:
        body = soup.find('body')
        if body:
            content = body.get_text(separator='\n', strip=True)

    # Clean up: remove excessive whitespace
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    content = '\n'.join(lines)

    # Truncate if too long
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS] + "\n\n[Content truncated - page too long]"

    if not content:
        return f"Could not extract text content from {url}. The page might be empty or use JavaScript rendering."

    # Build result with metadata
    result_parts = [f"**Title:** {title_text}"]

    if metadata.get("author"):
        result_parts.append(f"**Author:** {metadata['author']}")
    if metadata.get("date"):
        result_parts.append(f"**Date:** {metadata['date']}")
    if metadata.get("description"):
        result_parts.append(f"**Description:** {metadata['description']}")

    result_parts.append(f"\n**Content:**\n{content}")

    return "\n".join(result_parts)


@async_retry_on_error(
    max_retries=2,
    delay=1.0,
    exceptions=(aiohttp.ClientError, asyncio.TimeoutError)
)
async def fetch_url_content(url: str) -> str:
    """
    Fetch and extract readable content from a URL (HTML or PDF).

    Args:
        url: The web page or PDF URL to fetch

    Returns:
        Extracted text content with metadata, or error message.
    """
    try:
        headers = {"User-Agent": DEFAULT_USER_AGENT}

        # Fetch the content
        session = await get_aiohttp_session()
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=DEFAULT_HTTP_TIMEOUT)) as resp:
            resp.raise_for_status()

            # Check content type to determine how to parse
            content_type = resp.headers.get("Content-Type", "").lower()

            if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                # Handle PDF — read full bytes
                content_bytes = await resp.read()
                return _extract_pdf_content(content_bytes, url)
            else:
                # Handle HTML — read text
                html_text = await resp.text()
                return _extract_html_content(html_text, url)

    except asyncio.TimeoutError:
        return f"Request timed out while fetching {url}"
    except aiohttp.ClientResponseError as e:
        return f"HTTP error fetching {url}: {e.status}"
    except aiohttp.ClientError as e:
        return f"Error fetching URL: {str(e)}"
    except Exception as e:
        return f"Error processing content: {str(e)}"


# Create the LangChain Tool wrapper
url_tool = Tool(
    name="fetch_url",
    func=make_sync(fetch_url_content),
    coroutine=fetch_url_content,
    description=(
        "Read the TEXT CONTENT of a web page or PDF at a specific URL. Returns the "
        "main article text, title, and metadata as plain text."
        "\n\nUSE FOR:"
        "\n- Reading a full article/blog post from a URL"
        "\n- Following up on a link from web_search results"
        "\n- Reading simple PDFs (papers, reports)"
        "\n\nDO NOT USE FOR:"
        "\n- Extracting TABLES, LISTS, or LINKS (use web_scraper — it returns structured data)"
        "\n- Complex multi-column PDFs (use pdf_reader — it handles complex layouts better)"
        "\n- Searching for pages (use web_search first, then fetch_url on results)"
        "\n\nEXAMPLES: 'https://example.com/article', 'https://arxiv.org/pdf/1234.pdf'"
        "\n\nRULE: Need to READ a page as text? -> fetch_url. Need STRUCTURED DATA from it? -> web_scraper."
    )
)
