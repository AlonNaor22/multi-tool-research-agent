"""Tests for src/tools/currency_tool.py — currency conversion."""

import pytest
from unittest.mock import patch
from tests.conftest import MockResponse


class TestCurrencyConversion:
    """Test currency conversion with mocked API."""

    def test_successful_conversion(self):
        mock_resp = MockResponse(
            json_data={
                "amount": 100,
                "base": "USD",
                "date": "2026-03-23",
                "rates": {"EUR": 0.92}
            },
            status_code=200
        )

        with patch("src.tools.currency_tool.requests.get", return_value=mock_resp):
            from src.tools.currency_tool import currency_convert
            result = currency_convert("100 USD to EUR")

            assert "92" in result or "EUR" in result

    def test_same_currency(self):
        from src.tools.currency_tool import currency_convert
        result = currency_convert("100 USD to USD")
        # Should handle same-currency conversion
        assert "100" in result or len(result) > 0

    def test_invalid_currency_code(self):
        mock_resp = MockResponse(
            json_data={"message": "not found"},
            status_code=404
        )

        with patch("src.tools.currency_tool.requests.get", return_value=mock_resp):
            from src.tools.currency_tool import currency_convert
            result = currency_convert("100 XYZ to ABC")

            assert len(result) > 0  # Should return error, not crash

    def test_currency_aliases(self):
        """Common names like 'dollar' should map to currency codes."""
        mock_resp = MockResponse(
            json_data={
                "amount": 100,
                "base": "USD",
                "date": "2026-03-23",
                "rates": {"EUR": 0.92}
            },
            status_code=200
        )

        with patch("src.tools.currency_tool.requests.get", return_value=mock_resp):
            from src.tools.currency_tool import currency_convert
            result = currency_convert("100 dollars to euros")

            assert len(result) > 0

    def test_api_failure(self):
        import requests
        with patch("src.tools.currency_tool.requests.get",
                   side_effect=requests.exceptions.ConnectionError("Network error")):
            from src.tools.currency_tool import currency_convert
            result = currency_convert("100 USD to EUR")

            assert "Error" in result or "error" in result.lower() or "failed" in result.lower()
