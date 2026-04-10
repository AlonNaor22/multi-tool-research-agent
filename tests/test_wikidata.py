"""Tests for src/tools/wikidata_tool.py -- Wikidata knowledge base queries."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tests.conftest import AsyncMockResponse


SPARQL_ENTITY_RESPONSE = {
    "results": {
        "bindings": [
            {
                "item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q937"},
                "itemLabel": {"type": "literal", "value": "Albert Einstein"},
                "itemDescription": {"type": "literal", "value": "German-born physicist"},
                "property": {"type": "uri", "value": "http://www.wikidata.org/entity/P569"},
                "propertyLabel": {"type": "literal", "value": "date of birth"},
                "value": {"type": "literal", "value": "1879-03-14"},
                "valueLabel": {"type": "literal", "value": "14 March 1879"},
            },
            {
                "item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q937"},
                "itemLabel": {"type": "literal", "value": "Albert Einstein"},
                "itemDescription": {"type": "literal", "value": "German-born physicist"},
                "property": {"type": "uri", "value": "http://www.wikidata.org/entity/P27"},
                "propertyLabel": {"type": "literal", "value": "country of citizenship"},
                "value": {"type": "uri", "value": "http://www.wikidata.org/entity/Q30"},
                "valueLabel": {"type": "literal", "value": "United States of America"},
            },
        ]
    }
}

SPARQL_SEARCH_RESPONSE = {
    "results": {
        "bindings": [
            {
                "item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q937"},
                "itemLabel": {"type": "literal", "value": "Albert Einstein"},
                "itemDescription": {"type": "literal", "value": "German-born physicist"},
            },
        ]
    }
}


class TestWikidataQuery:
    """Test Wikidata queries with mocked SPARQL endpoint."""

    async def test_entity_lookup(self):
        mock_resp = AsyncMockResponse(json_data=SPARQL_ENTITY_RESPONSE, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.wikidata_tool import wikidata, _cache
            _cache.clear()
            result = await wikidata("Albert Einstein")

            assert "date of birth" in result
            assert "14 March 1879" in result

    async def test_search_mode(self):
        mock_resp = AsyncMockResponse(json_data=SPARQL_SEARCH_RESPONSE, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.wikidata_tool import wikidata, _cache
            _cache.clear()
            result = await wikidata("search: Einstein")

            assert "Albert Einstein" in result

    async def test_sparql_mode(self):
        mock_resp = AsyncMockResponse(json_data=SPARQL_SEARCH_RESPONSE, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.wikidata_tool import wikidata, _cache
            _cache.clear()
            result = await wikidata("sparql: SELECT ?item WHERE { ?item rdfs:label 'test'@en }")

            assert "SPARQL Results" in result or "no results" in result.lower()

    async def test_no_results_falls_back_to_search(self):
        empty_resp = AsyncMockResponse(json_data={"results": {"bindings": []}}, status=200)
        search_resp = AsyncMockResponse(json_data=SPARQL_SEARCH_RESPONSE, status=200)

        mock_session = MagicMock()
        # First call returns empty (entity lookup), second returns results (search)
        mock_session.get.side_effect = [empty_resp, search_resp]

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.wikidata_tool import wikidata, _cache
            _cache.clear()
            result = await wikidata("Einstein")

            assert "Albert Einstein" in result

    async def test_handles_request_error(self):
        import aiohttp
        mock_session = MagicMock()
        mock_session.get.side_effect = aiohttp.ClientError("failed")

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.wikidata_tool import wikidata, _cache
            _cache.clear()
            result = await wikidata("test")

            assert "Error" in result

    async def test_empty_query(self):
        from src.tools.wikidata_tool import wikidata
        result = await wikidata("")
        assert "Error" in result

    async def test_help_command(self):
        from src.tools.wikidata_tool import wikidata
        result = await wikidata("help")
        assert "SPARQL" in result

    async def test_caching(self):
        mock_resp = AsyncMockResponse(json_data=SPARQL_ENTITY_RESPONSE, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.wikidata_tool import wikidata, _cache
            _cache.clear()
            await wikidata("Albert Einstein")
            await wikidata("Albert Einstein")  # Should hit cache

            assert mock_session.get.call_count == 1
