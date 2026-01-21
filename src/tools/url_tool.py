"""URL content fetcher tool for the research agent.

Fetches and extracts readable text content from web pages.
Uses requests for HTTP and BeautifulSoup for HTML parsing.
No API key required.

This gives the agent ability to read specific web pages.
"""

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import Tool
from src.utils import retry_on_error


@retry_on_error(
    max_retries=2,
    delay=1.0,
    exceptions=(requests.exceptions.Timeout, requests.exceptions.ConnectionError)
)
def fetch_url_content(url: str) -> str:
    """
    Fetch and extract readable text content from a URL.

    Args:
        url: The web page URL to fetch (e.g., "https://example.com/article")

    Returns:
        Extracted text content from the page, or error message.
    """
    try:
        # Set headers to mimic a browser (some sites block requests without this)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Fetch the page
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Raise exception for 4xx/5xx status codes

        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements (we don't want JavaScript/CSS code)
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()

        # Extract text from the page
        # Get text from common content containers first
        content = ""

        # Try to find main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')

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

        # Truncate if too long (to fit in context window)
        max_chars = 4000
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[Content truncated - page too long]"

        if not content:
            return f"Could not extract text content from {url}. The page might be empty or use JavaScript rendering."

        # Get page title if available
        title = soup.find('title')
        title_text = title.get_text(strip=True) if title else "No title"

        return f"**Page Title:** {title_text}\n\n**Content:**\n{content}"

    except requests.exceptions.Timeout:
        return f"Request timed out while fetching {url}"
    except requests.exceptions.HTTPError as e:
        return f"HTTP error fetching {url}: {e.response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {str(e)}"
    except Exception as e:
        return f"Error processing page content: {str(e)}"


# Create the LangChain Tool wrapper
url_tool = Tool(
    name="fetch_url",
    func=fetch_url_content,
    description=(
        "Fetch and read the content of a specific web page URL. Use this when you "
        "have a specific URL and need to read its contents, or when web_search "
        "returns a relevant link that you want to explore in detail. "
        "Input should be a complete URL (e.g., 'https://example.com/article'). "
        "Returns the page title and main text content."
    )
)
