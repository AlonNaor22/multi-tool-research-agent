"""Wolfram Alpha tool for the research agent.

Wolfram Alpha is a "computational knowledge engine" that can:
- Solve math equations (algebra, calculus, etc.)
- Answer factual questions (population, distances, etc.)
- Convert units
- Compute statistics
- And much more!

KEY CONCEPT: API Integration
----------------------------
We're using Wolfram Alpha's "Short Answers API" which returns a simple text response.
This is different from their Full Results API which returns structured data.

The API is simple: just a GET request with your AppID and query.
Example: https://api.wolframalpha.com/v1/result?appid=XXX&i=population+of+france
"""

import requests
from urllib.parse import quote
from langchain_core.tools import Tool
from src.utils import retry_on_error
from config import WOLFRAM_ALPHA_APP_ID


# Wolfram Alpha Short Answers API endpoint
WOLFRAM_API_URL = "https://api.wolframalpha.com/v1/result"


@retry_on_error(max_retries=2, delay=1.0, exceptions=(Exception,))
def query_wolfram_alpha(query: str) -> str:
    """
    Query Wolfram Alpha and return the answer.

    HOW THE API WORKS:
    ------------------
    1. We URL-encode the query (spaces become %20, etc.)
    2. We send a GET request with our AppID and the query
    3. Wolfram Alpha returns a plain text answer

    The Short Answers API is perfect for our use case because:
    - Simple text response (no parsing needed)
    - Fast response time
    - Works for most common queries

    Args:
        query: The question or calculation to send to Wolfram Alpha
               Examples: "integrate x^2", "population of Japan", "150 USD to EUR"

    Returns:
        The answer as text, or an error message.
    """
    # Check if API key is configured
    if not WOLFRAM_ALPHA_APP_ID:
        return "Error: Wolfram Alpha API key not configured. Add WOLFRAM_ALPHA_APP_ID to your .env file."

    try:
        # Build the API request
        # quote() URL-encodes the query string
        params = {
            "appid": WOLFRAM_ALPHA_APP_ID,
            "i": query  # 'i' stands for 'input'
        }

        # Make the request
        # timeout=10 means we wait max 10 seconds for a response
        response = requests.get(
            WOLFRAM_API_URL,
            params=params,
            timeout=10
        )

        # Check for errors
        # Wolfram Alpha returns specific error messages in the response body
        if response.status_code == 200:
            answer = response.text.strip()

            # Wolfram returns specific messages for issues
            if answer == "Wolfram|Alpha did not understand your input":
                return f"Wolfram Alpha couldn't understand the query: '{query}'. Try rephrasing it."
            elif answer == "No short answer available":
                return f"Wolfram Alpha has data on this but no short answer. Query: '{query}'"
            else:
                return f"Wolfram Alpha: {answer}"

        elif response.status_code == 403:
            return "Error: Invalid Wolfram Alpha API key."
        elif response.status_code == 501:
            return f"Wolfram Alpha couldn't process this query: '{query}'"
        else:
            return f"Wolfram Alpha API error (status {response.status_code})"

    except requests.exceptions.Timeout:
        return "Error: Wolfram Alpha request timed out. Try again."
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Wolfram Alpha: {str(e)}"


# Create the LangChain Tool wrapper
wolfram_tool = Tool(
    name="wolfram_alpha",
    func=query_wolfram_alpha,
    description=(
        "Query Wolfram Alpha for computational knowledge. Use this for: "
        "mathematical calculations (integrals, derivatives, equations), "
        "unit conversions (e.g., '100 miles to km'), "
        "scientific data (e.g., 'atomic weight of gold'), "
        "factual questions (e.g., 'distance from Earth to Mars'), "
        "statistics and data lookups. "
        "Input should be a natural language query or mathematical expression. "
        "Examples: 'solve x^2 + 2x - 8 = 0', 'calories in an apple', "
        "'GDP of Germany', 'convert 72 fahrenheit to celsius'"
    )
)
