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

import asyncio
import aiohttp
from src.utils import async_retry_on_error, async_fetch, create_tool
from config import WOLFRAM_ALPHA_APP_ID


# Wolfram Alpha Short Answers API endpoint
WOLFRAM_API_URL = "https://api.wolframalpha.com/v1/result"


@async_retry_on_error(max_retries=2, delay=1.0, exceptions=(Exception,))
async def query_wolfram_alpha(query: str) -> str:
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

        answer = await async_fetch(
            WOLFRAM_API_URL, params=params, headers={}, timeout=10,
            response_type="text",
        )
        answer = answer.strip()

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

    except aiohttp.ClientResponseError as e:
        if e.status == 403:
            return "Error: Invalid Wolfram Alpha API key."
        elif e.status == 501:
            return f"Wolfram Alpha couldn't process: '{query}'. Try a different phrasing."
        else:
            return f"Wolfram Alpha API error (status {e.status})"
    except asyncio.TimeoutError:
        return "Error: Wolfram Alpha request timed out. Try again."
    except aiohttp.ClientError as e:
        return f"Error connecting to Wolfram Alpha: {str(e)}"


# Create the LangChain Tool wrapper
wolfram_tool = create_tool(
    "wolfram_alpha",
    query_wolfram_alpha,
    "Look up REFERENCE DATA from Wolfram Alpha's knowledge base: physical constants, "
    "scientific properties, nutritional data, and measurements that require an external database. "
    "\n\nUSE FOR (data lookups — things you can't compute yourself):"
    "\n- Scientific constants: 'speed of light in m/s', 'atomic weight of gold'"
    "\n- Physical properties: 'boiling point of ethanol', 'density of iron'"
    "\n- Nutritional data: 'calories in an apple', 'protein in 100g beef'"
    "\n- Astronomical data: 'distance Earth to Mars', 'diameter of Jupiter'"
    "\n- Geographic measurements: 'height of Mount Everest', 'depth of Pacific Ocean'"
    "\n\nDO NOT USE FOR:"
    "\n- Math calculations (use calculator — it handles arithmetic, algebra, trig)"
    "\n- Equations or symbolic math (use equation_solver)"
    "\n- Population, GDP, or entity facts (use wikidata — it has structured entity data)"
    "\n- Explanations or history (use wikipedia)"
    "\n- Current events (use web_search)"
    "\n\nRULE: Need a SCIENTIFIC CONSTANT or PHYSICAL PROPERTY? -> Wolfram. "
    "Need ENTITY FACTS (population, dates)? -> Wikidata. Need an EXPLANATION? -> Wikipedia.",
)
