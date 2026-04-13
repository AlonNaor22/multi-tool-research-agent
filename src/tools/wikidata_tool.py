"""Wikidata knowledge base tool — queries structured facts via SPARQL."""

import asyncio
import aiohttp
from typing import Dict, List

from langchain_core.tools import tool
from src.utils import async_retry_on_error, async_fetch, cached_tool, safe_tool_call, require_input
from src.constants import (
    WIKIDATA_SPARQL_URL,
    DEFAULT_HTTP_TIMEOUT,
)

# ─── Module overview ───────────────────────────────────────────────
# Queries structured entity facts from Wikidata via SPARQL.
# Supports entity lookup, search, and raw SPARQL queries.
# ───────────────────────────────────────────────────────────────────

# Sends a SPARQL query to the Wikidata endpoint and returns parsed bindings.
@async_retry_on_error(max_retries=2, delay=2.0)
async def _run_sparql(sparql_query: str) -> List[Dict]:
    """Execute a SPARQL query against Wikidata and return results."""
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "ResearchAgent/1.0",
    }

    data = await async_fetch(
        WIKIDATA_SPARQL_URL,
        params={"query": sparql_query},
        headers=headers,
        timeout=DEFAULT_HTTP_TIMEOUT,
    )

    bindings = data.get("results", {}).get("bindings", [])

    results = []
    for binding in bindings:
        row = {}
        for key, val in binding.items():
            row[key] = val.get("value", "")
        results.append(row)

    return results


# Builds a SPARQL query that finds an entity by exact name and returns its properties.
def _entity_lookup_sparql(entity_name: str, limit: int = 10) -> str:
    """Build a SPARQL query that finds an entity and returns key facts."""
    return f"""
SELECT ?item ?itemLabel ?itemDescription ?property ?propertyLabel ?value ?valueLabel WHERE {{
  ?item rdfs:label "{entity_name}"@en .
  ?item ?prop ?statement .
  ?statement ?ps ?value .
  ?property wikibase:claim ?prop .
  ?property wikibase:statementProperty ?ps .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT {limit}
"""


# Builds a SPARQL query that fuzzy-searches for entities by partial name.
def _search_entities_sparql(search_term: str, limit: int = 5) -> str:
    """Build a SPARQL query that searches for entities by partial name match."""
    return f"""
SELECT ?item ?itemLabel ?itemDescription WHERE {{
  SERVICE wikibase:mwapi {{
    bd:serviceParam wikibase:endpoint "www.wikidata.org";
                    wikibase:api "EntitySearch";
                    mwapi:search "{search_term}";
                    mwapi:language "en".
    ?item wikibase:apiOutputItem mwapi:item.
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT {limit}
"""


# Formats SPARQL property/value rows into a human-readable fact list.
def format_entity_facts(results: List[Dict], entity: str) -> str:
    """Format entity facts from SPARQL results."""
    if not results:
        return f"No Wikidata facts found for '{entity}'."

    lines = [f"Wikidata Facts for '{entity}':", ""]

    seen = set()
    for row in results:
        prop = row.get("propertyLabel", "")
        val = row.get("valueLabel", "")

        if not prop or not val:
            continue

        key = f"{prop}:{val}"
        if key in seen:
            continue
        seen.add(key)

        if val.startswith("http"):
            continue

        lines.append(f"  {prop}: {val}")

    if len(lines) <= 2:
        return f"No readable facts found for '{entity}'. The entity may not exist or uses a different name."

    return "\n".join(lines)


# Formats entity search results with label, description, and Wikidata URL.
def format_search_results(results: List[Dict], query: str) -> str:
    """Format entity search results."""
    if not results:
        return f"No Wikidata entities found matching '{query}'."

    lines = [f"Wikidata Entities matching '{query}':", ""]

    for i, row in enumerate(results, 1):
        label = row.get("itemLabel", "Unknown")
        desc = row.get("itemDescription", "")
        item_url = row.get("item", "")
        lines.append(f"{i}. {label}")
        if desc:
            lines.append(f"   {desc}")
        if item_url:
            lines.append(f"   {item_url}")
        lines.append("")

    return "\n".join(lines)


# Takes an entity name, "search: term", or "sparql: query" string.
# Routes to the appropriate SPARQL query mode and returns formatted results.
@safe_tool_call("querying Wikidata")
async def wikidata(input_str: str) -> str:
    """Look up STRUCTURED ENTITY FACTS from Wikidata — the database behind Wikipedia. Use when you need a specific property of a known entity (country, person, city, etc.).

USE FOR:
- Entity properties: population, GDP, area, coordinates, founding date
- Relationships: 'who is the president of France', 'capital of Japan'
- Classifications: 'what type of animal is a platypus'
- Cross-referencing: verify a fact from another source

DO NOT USE FOR:
- Explanations or history (use wikipedia)
- Scientific constants or physical properties (use wolfram_alpha)
- Current events or recent changes (use web_search)

FORMAT: 'entity name', 'search: term', 'sparql: SPARQL query'

RULE: Need a FACT about an ENTITY? -> Wikidata. Need an EXPLANATION? -> Wikipedia. Need a SCIENTIFIC VALUE? -> Wolfram."""
    err = require_input(input_str, "query")
    if err:
        return err

    if input_str.strip().lower() in ("help", "?"):
        return _get_help()

    return await _wikidata_cached_fetch(input_str.strip())


# Cached inner function that dispatches SPARQL, search, or entity lookups.
@cached_tool("wikidata")
async def _wikidata_cached_fetch(input_str: str) -> str:
    """Inner cacheable function for wikidata_query."""
    # Raw SPARQL mode
    if input_str.lower().startswith("sparql:"):
        sparql = input_str[7:].strip()
        results = await _run_sparql(sparql)
        if not results:
            return "SPARQL query returned no results."
        lines = [f"SPARQL Results ({len(results)} rows):", ""]
        for i, row in enumerate(results[:20], 1):
            parts = [f"{k}: {v}" for k, v in row.items()]
            lines.append(f"{i}. {' | '.join(parts)}")
        if len(results) > 20:
            lines.append(f"[... {len(results) - 20} more rows ...]")
        return "\n".join(lines)

    # Search mode
    if input_str.lower().startswith("search:"):
        search_term = input_str[7:].strip()
        sparql = _search_entities_sparql(search_term)
        results = await _run_sparql(sparql)
        return format_search_results(results, search_term)

    # Default: entity fact lookup
    entity = input_str
    sparql = _entity_lookup_sparql(entity)
    results = await _run_sparql(sparql)

    if not results:
        sparql = _search_entities_sparql(entity)
        results = await _run_sparql(sparql)
        return format_search_results(results, entity)

    return format_entity_facts(results, entity)


def _get_help() -> str:
    """Return help text."""
    return """Wikidata Knowledge Base Help:

FORMAT:
  Albert Einstein               - Look up facts about an entity
  search: quantum physics       - Search for matching entities
  sparql: SELECT ?x WHERE ...   - Run a raw SPARQL query

RETURNS:
  - Structured facts (dates, populations, coordinates, relationships)
  - Entity descriptions and classifications
  - Links to Wikidata pages

USE CASES:
  - Get precise dates, populations, geographic data
  - Find relationships between entities
  - Verify facts from other sources
  - Access multilingual structured data

TIPS:
  - Use exact entity names for best results (e.g., "Paris" not "paris france")
  - Use "search:" prefix if you're unsure of the exact name
  - Raw SPARQL gives you full power over Wikidata's knowledge graph

EXAMPLES:
  "Tokyo"
  "search: programming languages"
  "Marie Curie"
  "sparql: SELECT ?city ?pop WHERE { ?city wdt:P31 wd:Q515 . ?city wdt:P1082 ?pop } LIMIT 5" """


# Expose cache for tests (_wikidata_cached_fetch._cache)
_cache = _wikidata_cached_fetch._cache

wikidata_tool = tool(wikidata)
