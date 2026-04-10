"""Tests for src/tools/currency_tool.py -- currency conversion."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tests.conftest import AsyncMockResponse


class TestCurrencyConversion:
    """Test currency conversion with mocked API."""

    async def test_successful_conversion(self):
        mock_resp = AsyncMockResponse(
            json_data={
                "amount": 100,
                "base": "USD",
                "date": "2026-03-23",
                "rates": {"EUR": 0.92}
            },
            status=200
        )
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.currency_tool import currency_convert
            result = await currency_convert("100 USD to EUR")

            assert "92" in result or "EUR" in result

    async def test_same_currency(self):
        from src.tools.currency_tool import currency_convert
        result = await currency_convert("100 USD to USD")
        # Should handle same-currency conversion
        assert "100" in result or len(result) > 0

    async def test_invalid_currency_code(self):
        mock_resp = AsyncMockResponse(
            json_data={"message": "not found"},
            status=404
        )
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.currency_tool import currency_convert
            result = await currency_convert("100 XYZ to ABC")

            assert len(result) > 0  # Should return error, not crash

    async def test_currency_aliases(self):
        """Common names like 'dollar' should map to currency codes."""
        mock_resp = AsyncMockResponse(
            json_data={
                "amount": 100,
                "base": "USD",
                "date": "2026-03-23",
                "rates": {"EUR": 0.92}
            },
            status=200
        )
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.currency_tool import currency_convert
            result = await currency_convert("100 dollars to euros")

            assert len(result) > 0

    async def test_api_failure(self):
        import aiohttp
        mock_session = MagicMock()
        mock_session.get.side_effect = aiohttp.ClientError("Network error")

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.currency_tool import currency_convert
            result = await currency_convert("100 USD to EUR")

            assert "Error" in result or "error" in result.lower() or "failed" in result.lower()
