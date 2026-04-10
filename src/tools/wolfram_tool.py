"""Wolfram Alpha tool — queries the Short Answers API for precise, quantitative data."""

import aiohttp
from langchain_core.tools import tool
from src.utils import async_retry_on_error, async_fetch, safe_tool_call
from config import WOLFRAM_ALPHA_APP_ID


# Wolfram Alpha Short Answers API endpoint
WOLFRAM_API_URL = "https://api.wolframalpha.com/v1/result"


@safe_tool_call("querying Wolfram Alpha")
@async_retry_on_error(max_retries=2, delay=1.0, exceptions=(Exception,))
async def wolfram_alpha(query: str) -> str:
    """Look up REFERENCE DATA from Wolfram Alpha's knowledge base: physical constants, scientific properties, nutritional data, and measurements that require an external database.

USE FOR (data lookups — things you can't compute yourself):
- Scientific constants: 'speed of light in m/s', 'atomic weight of gold'
- Physical properties: 'boiling point of ethanol', 'density of iron'
- Nutritional data: 'calories in an apple', 'protein in 100g beef'
- Astronomical data: 'distance Earth to Mars', 'diameter of Jupiter'
- Geographic measurements: 'height of Mount Everest', 'depth of Pacific Ocean'

DO NOT USE FOR:
- Math calculations (use calculator — it handles arithmetic, algebra, trig)
- Equations or symbolic math (use equation_solver)
- Population, GDP, or entity facts (use wikidata — it has structured entity data)
- Explanations or history (use wikipedia)
- Current events (use web_search)

RULE: Need a SCIENTIFIC CONSTANT or PHYSICAL PROPERTY? -> Wolfram. Need ENTITY FACTS (population, dates)? -> Wikidata. Need an EXPLANATION? -> Wikipedia."""
    if not WOLFRAM_ALPHA_APP_ID:
        return (
            "Error: Wolfram Alpha API key not configured. "
            "Add WOLFRAM_ALPHA_APP_ID to your .env file. "
            "Get a free key at: https://developer.wolframalpha.com/"
        )

    params = {
        "appid": WOLFRAM_ALPHA_APP_ID,
        "i": query
    }

    answer = await async_fetch(
        WOLFRAM_API_URL, params=params, headers={}, timeout=10,
        response_type="text",
    )
    answer = answer.strip()

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


wolfram_tool = tool(wolfram_alpha)
