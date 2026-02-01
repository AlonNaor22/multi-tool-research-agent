"""Wolfram Alpha tool for the research agent.

Wolfram Alpha is a "computational knowledge engine" for PRECISE, QUANTITATIVE DATA.
Use this when you need exact numbers, measurements, or scientific values.

WHEN TO USE THIS TOOL (precise/quantitative answers):
----------------------------------------------------
- Nutritional data: "calories in a banana", "protein in 100g chicken"
- Scientific constants: "speed of light in m/s", "atomic mass of gold"
- Physical measurements: "height of Mount Everest", "depth of Pacific Ocean"
- Astronomical data: "distance from Earth to Mars", "diameter of Jupiter"
- Chemical properties: "boiling point of ethanol", "density of iron"
- Precise statistics: "population of Tokyo", "GDP of Germany in USD"

WHEN NOT TO USE THIS TOOL:
-------------------------
- General knowledge/history/context -> use 'wikipedia'
- Current events/news -> use 'web_search' or 'news_search'
- Math calculations -> use 'calculator'
- Unit conversions -> use 'unit_converter'
- Solving equations -> use 'equation_solver'

RULE OF THUMB:
- Need a specific NUMBER or MEASUREMENT? -> Wolfram Alpha
- Need an EXPLANATION or CONTEXT? -> Wikipedia
- Need CURRENT or RECENT info? -> Web Search

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
        "Get PRECISE NUMERICAL DATA and exact measurements. Use when you need "
        "a specific number, not an explanation. "
        "\n\nUSE FOR (exact values):"
        "\n- Nutritional data: 'calories in an apple', 'protein in 100g beef'"
        "\n- Scientific constants: 'speed of light in m/s', 'atomic weight of gold'"
        "\n- Measurements: 'height of Mount Everest in meters', 'depth of Pacific Ocean'"
        "\n- Astronomical: 'distance Earth to Mars in km', 'diameter of Jupiter'"
        "\n- Chemical: 'boiling point of ethanol', 'density of iron'"
        "\n- Statistics: 'population of Tokyo', 'GDP of France in USD'"
        "\n\nDO NOT USE FOR:"
        "\n- Explanations/history/context (use wikipedia)"
        "\n- Current events (use web_search)"
        "\n- Math (use calculator) or equations (use equation_solver)"
        "\n\nRULE: Need a NUMBER? -> Wolfram. Need an EXPLANATION? -> Wikipedia."
    )
)
