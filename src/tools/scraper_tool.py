"""Web scraper tool for the research agent.

Extracts structured data (tables, lists, links, headings) from web pages.
Goes beyond fetch_url by returning structured data instead of raw text.
Uses BeautifulSoup for parsing. No API key required.

Features:
- Table extraction as formatted markdown
- List extraction (ordered and unordered)
- Link extraction with text and URLs
- Heading extraction for page structure
- CSS selector-based custom extraction
"""

import json
import aiohttp
from bs4 import BeautifulSoup
from langchain_core.tools import Tool
from src.utils import async_retry_on_error, get_aiohttp_session, make_sync
from src.constants import DEFAULT_USER_AGENT, DEFAULT_HTTP_TIMEOUT, DEFAULT_MAX_CONTENT_CHARS


@async_retry_on_error(max_retries=2, delay=1.0, exceptions=(Exception,))
async def _fetch_html(url: str) -> str:
    """Fetch raw HTML from a URL."""
    session = await get_aiohttp_session()
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    async with session.get(
        url,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=DEFAULT_HTTP_TIMEOUT),
    ) as resp:
        resp.raise_for_status()
        return await resp.text()


def _extract_tables(soup: BeautifulSoup, max_tables: int = 5) -> str:
    """Extract tables as markdown."""
    tables = soup.find_all("table")
    if not tables:
        return ""

    results = []
    for idx, table in enumerate(tables[:max_tables], 1):
        rows = table.find_all("tr")
        if not rows:
            continue

        table_data = []
        for row in rows:
            cells = row.find_all(["th", "td"])
            table_data.append([cell.get_text(strip=True) for cell in cells])

        if not table_data:
            continue

        # Build markdown table
        max_cols = max(len(row) for row in table_data)
        # Pad rows to same length
        for row in table_data:
            while len(row) < max_cols:
                row.append("")

        lines = []
        # Header row
        lines.append("| " + " | ".join(table_data[0]) + " |")
        lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        # Data rows
        for row in table_data[1:]:
            lines.append("| " + " | ".join(row) + " |")

        results.append(f"**Table {idx}:**\n" + "\n".join(lines))

    return "\n\n".join(results)


def _extract_lists(soup: BeautifulSoup, max_lists: int = 5) -> str:
    """Extract ordered and unordered lists."""
    all_lists = soup.find_all(["ul", "ol"])
    if not all_lists:
        return ""

    results = []
    for idx, lst in enumerate(all_lists[:max_lists], 1):
        items = lst.find_all("li", recursive=False)
        if not items:
            continue

        is_ordered = lst.name == "ol"
        lines = []
        for i, item in enumerate(items, 1):
            text = item.get_text(strip=True)[:200]
            prefix = f"{i}." if is_ordered else "-"
            lines.append(f"  {prefix} {text}")

        if lines:
            results.append(f"**List {idx}:**\n" + "\n".join(lines))

    return "\n\n".join(results)


def _extract_links(soup: BeautifulSoup, max_links: int = 20) -> str:
    """Extract links with text and URLs."""
    links = soup.find_all("a", href=True)
    if not links:
        return ""

    results = []
    seen = set()
    for link in links:
        href = link["href"]
        text = link.get_text(strip=True)
        if not text or not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        if href in seen:
            continue
        seen.add(href)
        results.append(f"- [{text[:80]}]({href})")
        if len(results) >= max_links:
            break

    return "**Links:**\n" + "\n".join(results) if results else ""


def _extract_headings(soup: BeautifulSoup) -> str:
    """Extract page structure from headings."""
    headings = soup.find_all(["h1", "h2", "h3", "h4"])
    if not headings:
        return ""

    lines = []
    for h in headings:
        level = int(h.name[1])
        indent = "  " * (level - 1)
        text = h.get_text(strip=True)[:100]
        lines.append(f"{indent}- {text}")

    return "**Page Structure:**\n" + "\n".join(lines)


async def scrape_webpage(query: str) -> str:
    """Scrape structured data from a web page.

    Input can be a URL string or JSON with options:
    - Simple: "https://example.com"
    - Advanced: {"url": "https://example.com", "extract": ["tables", "lists", "links", "headings"]}
    - CSS selector: {"url": "https://example.com", "selector": "div.main-content"}

    Extract options: tables, lists, links, headings, all (default)
    """
    # Parse input
    extract_types = ["tables", "lists", "links", "headings"]
    css_selector = None

    try:
        if query.strip().startswith("{"):
            options = json.loads(query)
            url = options.get("url", "")
            extract = options.get("extract", ["all"])
            css_selector = options.get("selector")
            if "all" not in extract:
                extract_types = extract
        else:
            url = query.strip()
    except json.JSONDecodeError:
        url = query.strip()

    if not url:
        return "Error: No URL provided."

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        html = await _fetch_html(url)
        soup = BeautifulSoup(html, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "noscript"]):
            tag.decompose()

        # Apply CSS selector if provided
        if css_selector:
            target = soup.select_one(css_selector)
            if target:
                soup = BeautifulSoup(str(target), "html.parser")
            else:
                return f"CSS selector '{css_selector}' not found on {url}"

        # Get page title
        title = soup.find("title")
        title_text = title.get_text(strip=True) if title else url

        sections = [f"**Scraped: {title_text}**\n**URL:** {url}\n"]

        if "headings" in extract_types:
            headings = _extract_headings(soup)
            if headings:
                sections.append(headings)

        if "tables" in extract_types:
            tables = _extract_tables(soup)
            if tables:
                sections.append(tables)

        if "lists" in extract_types:
            lists = _extract_lists(soup)
            if lists:
                sections.append(lists)

        if "links" in extract_types:
            links = _extract_links(soup)
            if links:
                sections.append(links)

        result = "\n\n".join(sections)

        if len(result) > DEFAULT_MAX_CONTENT_CHARS:
            result = result[:DEFAULT_MAX_CONTENT_CHARS] + "\n\n[Content truncated]"

        if len(sections) <= 1:
            return f"No structured data (tables, lists, links) found on {url}. Try fetch_url for raw text content."

        return result

    except Exception as e:
        return f"Error scraping {url}: {str(e)}"


scraper_tool = Tool(
    name="web_scraper",
    func=make_sync(scrape_webpage),
    coroutine=scrape_webpage,
    description=(
        "Extract structured data from web pages — tables, lists, links, and headings. "
        "Use this instead of fetch_url when you need organized data, not raw text."
        "\n\nUSE FOR:"
        "\n- Tables: statistics pages, comparison charts, data tables"
        "\n- Lists: product features, ranked items, requirements"
        "\n- Links: resource pages, directories, navigation structure"
        "\n- Page structure: understand how content is organized"
        "\n\nSIMPLE: 'https://example.com' (extracts all structured data)"
        "\n\nADVANCED: {\"url\": \"...\", \"extract\": [\"tables\", \"links\"]}"
        "\n\nCSS SELECTOR: {\"url\": \"...\", \"selector\": \"div.main-content\"}"
        "\n\nDO NOT USE FOR: raw text reading (use fetch_url), PDF files (use pdf_reader)"
    ),
)
