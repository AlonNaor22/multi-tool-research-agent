"""Wikidata knowledge base tool for the research agent.

Queries structured facts from Wikidata using the SPARQL endpoint.
Returns precise, structured data (dates, populations, coordinates,
relationships) that complements unstructured search results.

Free, no API key required.
"""

import re
import json
import requests
from typing import Dict, List, Optional
from langchain_core.tools import Tool

from src.utils import retry_on_error, TTLCache
from src.constants import (
    WIKIDATA_SPARQL_URL,
    DEFAULT_HTTP_TIMEOUT,
    DEFAULT_CACHE_TTL,
)

# Cache repeated queries for 5 minutes
_cache = TTLCache(ttl=DEFAULT_CACHE_TTL)


@retry_on_error(max_retries=2, delay=2.0)
def _run_sparql(sparql_query: str) -> List[Dict]:
    """Execute a SPARQL query against Wikidata and return results."""
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "ResearchAgent/1.0",
    }

    response = requests.get(
        WIKIDATA_SPARQL_URL,
        params={"query": sparql_query},
        headers=headers,
        timeout=DEFAULT_HTTP_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()
    bindings = data.get("results", {}).get("bindings", [])

    # Simplify bindings: extract just the values
    results = []
    for binding in bindings:
        row = {}
        for key, val in binding.items():
            row[key] = val.get("value", "")
        results.append(row)

    return results


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

        # Skip duplicates and internal/technical properties
        key = f"{prop}:{val}"
        if key in seen:
            continue
        seen.add(key)

        # Clean up Wikidata URLs to just show labels
        if val.startswith("http"):
            continue

        lines.append(f"  {prop}: {val}")

    if len(lines) <= 2:
        return f"No readable facts found for '{entity}'. The entity may not exist or uses a different name."

    return "\n".join(lines)


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


def wikidata_query(input_str: str) -> str:
    """
    Query Wikidata for structured facts about entities.

    Supports formats:
    - "Albert Einstein"        → look up facts about an entity
    - "search: quantum physics" → search for matching entities
    - "sparql: SELECT ..."     → run a raw SPARQL query

    Args:
        input_str: Entity name, search term, or SPARQL query

    Returns:
        Formatted facts or search results
    """
    input_str = input_str.strip()

    if not input_str:
        return "Error: Empty query"

    if input_str.lower() in ("help", "?"):
        return _get_help()

    cache_key = _cache.make_key("wikidata", input_str)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        # Raw SPARQL mode
        if input_str.lower().startswith("sparql:"):
            sparql = input_str[7:].strip()
            results = _run_sparql(sparql)
            if not results:
                output = "SPARQL query returned no results."
            else:
                # Format as a simple table
                lines = [f"SPARQL Results ({len(results)} rows):", ""]
                for i, row in enumerate(results[:20], 1):
                    parts = [f"{k}: {v}" for k, v in row.items()]
                    lines.append(f"{i}. {' | '.join(parts)}")
                if len(results) > 20:
                    lines.append(f"[... {len(results) - 20} more rows ...]")
                output = "\n".join(lines)
            _cache.set(cache_key, output)
            return output

        # Search mode
        if input_str.lower().startswith("search:"):
            search_term = input_str[7:].strip()
            sparql = _search_entities_sparql(search_term)
            results = _run_sparql(sparql)
            output = format_search_results(results, search_term)
            _cache.set(cache_key, output)
            return output

        # Default: entity fact lookup
        entity = input_str
        sparql = _entity_lookup_sparql(entity)
        results = _run_sparql(sparql)

        if not results:
            # Try searching instead of exact match
            sparql = _search_entities_sparql(entity)
            results = _run_sparql(sparql)
            output = format_search_results(results, entity)
        else:
            output = format_entity_facts(results, entity)

        _cache.set(cache_key, output)
        return output

    except requests.exceptions.RequestException as e:
        return f"Error querying Wikidata: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


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


# Create the LangChain Tool wrapper
wikidata_tool = Tool(
    name="wikidata",
    func=wikidata_query,
    description=(
        "Query Wikidata for precise, structured facts about any entity. "
        "\n\nFORMAT: 'entity name', 'search: term', 'sparql: SPARQL query'"
        "\n\nRETURNS: Structured facts — dates, populations, coordinates, relationships, classifications."
        "\n\nUSE FOR: Precise factual data (dates, numbers, relationships), verifying facts from other sources."
        "\n\nTIP: Use exact entity names. Use 'search:' if unsure of the exact name."
    )
)
