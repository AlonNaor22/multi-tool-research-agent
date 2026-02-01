"""Wolfram Alpha tool for the research agent.

Wolfram Alpha is a "computational knowledge engine" - use it as a LAST RESORT
for queries that other tools cannot handle.

WHEN TO USE THIS TOOL:
---------------------
- Real-world factual data: "calories in a banana", "population of Tokyo"
- Scientific constants: "speed of light", "atomic mass of gold"
- Geographic/astronomical data: "distance from Earth to Mars", "height of Everest"
- Economic data: "GDP of Germany", "minimum wage in California"
- Nutritional information: "protein in chicken breast"
- Historical dates: "when was the Eiffel Tower built"
- Comparisons: "compare population of France and Germany"

WHEN NOT TO USE THIS TOOL (use these instead):
---------------------------------------------
- Basic math calculations -> use 'calculator'
- Unit conversions -> use 'unit_converter'
- Solving equations (x + 2 = 5) -> use 'equation_solver'
- Current events/news -> use 'web_search' or 'news_search'
- Encyclopedia info -> use 'wikipedia'
- Academic papers -> use 'arxiv_search'

This tool queries Wolfram Alpha's Short Answers API for concise responses.
Requires WOLFRAM_ALPHA_APP_ID in your .env file.
"""

import requests
from langchain_core.tools import Tool
from src.utils import retry_on_error
from config import WOLFRAM_ALPHA_APP_ID


# Wolfram Alpha Short Answers API endpoint
WOLFRAM_API_URL = "https://api.wolframalpha.com/v1/result"


@retry_on_error(max_retries=2, delay=1.0, exceptions=(Exception,))
def query_wolfram_alpha(query: str) -> str:
    """
    Query Wolfram Alpha for factual/computational knowledge.

    This is best used for real-world data that requires Wolfram's
    curated knowledge base - things like nutritional info, scientific
    constants, geographic data, economic statistics, etc.

    Args:
        query: Natural language question or lookup
               Examples: "calories in an avocado", "height of Mount Fuji",
                        "GDP of Japan", "boiling point of ethanol"

    Returns:
        The answer as text, or an error message.
    """
    # Check if API key is configured
    if not WOLFRAM_ALPHA_APP_ID:
        return (
            "Error: Wolfram Alpha API key not configured. "
            "Add WOLFRAM_ALPHA_APP_ID to your .env file. "
            "Get a free key at: https://developer.wolframalpha.com/"
        )

    try:
        # Build the API request
        params = {
            "appid": WOLFRAM_ALPHA_APP_ID,
            "i": query
        }

        # Make the request with timeout
        response = requests.get(
            WOLFRAM_API_URL,
            params=params,
            timeout=10
        )

        # Handle response
        if response.status_code == 200:
            answer = response.text.strip()

            # Wolfram returns specific messages for issues
            if answer == "Wolfram|Alpha did not understand your input":
                return (
                    f"Wolfram Alpha couldn't understand: '{query}'. "
                    "Try rephrasing as a simple factual question."
                )
            elif answer == "No short answer available":
                return (
                    f"Wolfram Alpha has data on this but no short answer available. "
                    f"Query: '{query}'. Try being more specific."
                )
            else:
                return f"Wolfram Alpha: {answer}"

        elif response.status_code == 403:
            return "Error: Invalid Wolfram Alpha API key."
        elif response.status_code == 501:
            return f"Wolfram Alpha couldn't process: '{query}'. Try a different phrasing."
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
        "Query Wolfram Alpha for FACTUAL REAL-WORLD DATA. Use this as a LAST RESORT "
        "when other tools cannot answer the question. "
        "\n\nBEST FOR:"
        "\n- Nutritional data: 'calories in an apple', 'protein in eggs'"
        "\n- Scientific constants: 'speed of light', 'atomic weight of gold'"
        "\n- Geographic facts: 'population of Tokyo', 'height of Mount Everest'"
        "\n- Economic data: 'GDP of France', 'unemployment rate in US'"
        "\n- Astronomical data: 'distance to Mars', 'diameter of Jupiter'"
        "\n- Historical facts: 'when was the Eiffel Tower built'"
        "\n\nDO NOT USE FOR:"
        "\n- Math calculations (use calculator)"
        "\n- Unit conversions (use unit_converter)"
        "\n- Solving equations (use equation_solver)"
        "\n- Current news (use web_search)"
        "\n- General knowledge (use wikipedia)"
    )
)
