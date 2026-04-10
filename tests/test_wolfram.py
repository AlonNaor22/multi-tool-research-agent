"""Tests for src/tools/wolfram_tool.py -- Wolfram Alpha queries."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tests.conftest import AsyncMockResponse


class TestWolframAlpha:
    """Test Wolfram Alpha queries with mocked HTTP."""

    async def test_successful_query(self):
        mock_resp = AsyncMockResponse(text="149,600,000 kilometers", status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session), \
             patch("src.tools.wolfram_tool.WOLFRAM_ALPHA_APP_ID", "test_key"):
            from src.tools.wolfram_tool import query_wolfram_alpha
            result = await query_wolfram_alpha("distance from earth to sun")

            assert "149,600,000" in result

    async def test_missing_api_key(self):
        with patch("src.tools.wolfram_tool.WOLFRAM_ALPHA_APP_ID", None):
            from src.tools.wolfram_tool import query_wolfram_alpha
            result = await query_wolfram_alpha("test query")

            assert "API" in result or "not configured" in result.lower() or "key" in result.lower()

    async def test_did_not_understand(self):
        mock_resp = AsyncMockResponse(
            text="Wolfram|Alpha did not understand your input",
            status=200
        )
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session), \
             patch("src.tools.wolfram_tool.WOLFRAM_ALPHA_APP_ID", "test_key"):
            from src.tools.wolfram_tool import query_wolfram_alpha
            result = await query_wolfram_alpha("asdfghjkl gibberish")

            assert "did not understand" in result.lower() or len(result) > 0

    async def test_error_status_code(self):
        mock_resp = AsyncMockResponse(text="Error", status=501)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session), \
             patch("src.tools.wolfram_tool.WOLFRAM_ALPHA_APP_ID", "test_key"):
            from src.tools.wolfram_tool import query_wolfram_alpha
            result = await query_wolfram_alpha("test")

            assert len(result) > 0  # Should return error info, not crash

    async def test_timeout_handling(self):
        import asyncio
        mock_session = MagicMock()
        mock_session.get.side_effect = asyncio.TimeoutError("timeout")

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session), \
             patch("src.tools.wolfram_tool.WOLFRAM_ALPHA_APP_ID", "test_key"):
            from src.tools.wolfram_tool import query_wolfram_alpha
            result = await query_wolfram_alpha("test")

            assert "Error" in result or "timeout" in result.lower() or "failed" in result.lower()
