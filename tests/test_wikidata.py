"""Tests for src/tools/wikidata_tool.py — Wikidata knowledge base queries."""

import pytest
from unittest.mock import patch
from tests.conftest import MockResponse


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

    def test_entity_lookup(self):
        mock_resp = MockResponse(json_data=SPARQL_ENTITY_RESPONSE, status_code=200)

        with patch("src.tools.wikidata_tool.requests.get", return_value=mock_resp):
            from src.tools.wikidata_tool import wikidata_query, _cache
            _cache.clear()
            result = wikidata_query("Albert Einstein")

            assert "date of birth" in result
            assert "14 March 1879" in result

    def test_search_mode(self):
        mock_resp = MockResponse(json_data=SPARQL_SEARCH_RESPONSE, status_code=200)

        with patch("src.tools.wikidata_tool.requests.get", return_value=mock_resp):
            from src.tools.wikidata_tool import wikidata_query, _cache
            _cache.clear()
            result = wikidata_query("search: Einstein")

            assert "Albert Einstein" in result

    def test_sparql_mode(self):
        mock_resp = MockResponse(json_data=SPARQL_SEARCH_RESPONSE, status_code=200)

        with patch("src.tools.wikidata_tool.requests.get", return_value=mock_resp):
            from src.tools.wikidata_tool import wikidata_query, _cache
            _cache.clear()
            result = wikidata_query("sparql: SELECT ?item WHERE { ?item rdfs:label 'test'@en }")

            assert "SPARQL Results" in result or "no results" in result.lower()

    def test_no_results_falls_back_to_search(self):
        empty_resp = MockResponse(json_data={"results": {"bindings": []}}, status_code=200)
        search_resp = MockResponse(json_data=SPARQL_SEARCH_RESPONSE, status_code=200)

        # First call returns empty (entity lookup), second returns results (search)
        with patch("src.tools.wikidata_tool.requests.get",
                   side_effect=[empty_resp, search_resp]):
            from src.tools.wikidata_tool import wikidata_query, _cache
            _cache.clear()
            result = wikidata_query("Einstein")

            assert "Albert Einstein" in result

    def test_handles_request_error(self):
        import requests
        with patch("src.tools.wikidata_tool.requests.get",
                   side_effect=requests.exceptions.ConnectionError("failed")):
            from src.tools.wikidata_tool import wikidata_query, _cache
            _cache.clear()
            result = wikidata_query("test")

            assert "Error" in result

    def test_empty_query(self):
        from src.tools.wikidata_tool import wikidata_query
        result = wikidata_query("")
        assert "Error" in result

    def test_help_command(self):
        from src.tools.wikidata_tool import wikidata_query
        result = wikidata_query("help")
        assert "SPARQL" in result

    def test_caching(self):
        mock_resp = MockResponse(json_data=SPARQL_ENTITY_RESPONSE, status_code=200)

        with patch("src.tools.wikidata_tool.requests.get", return_value=mock_resp) as mock_get:
            from src.tools.wikidata_tool import wikidata_query, _cache
            _cache.clear()
            wikidata_query("Albert Einstein")
            wikidata_query("Albert Einstein")  # Should hit cache

            assert mock_get.call_count == 1
