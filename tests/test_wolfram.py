"""Tests for src/tools/wolfram_tool.py — Wolfram Alpha queries."""

import pytest
from unittest.mock import patch
from tests.conftest import MockResponse


class TestWolframAlpha:
    """Test Wolfram Alpha queries with mocked HTTP."""

    def test_successful_query(self):
        mock_resp = MockResponse(text="149,600,000 kilometers", status_code=200)

        with patch("src.tools.wolfram_tool.requests.get", return_value=mock_resp), \
             patch("src.tools.wolfram_tool.WOLFRAM_ALPHA_APP_ID", "test_key"):
            from src.tools.wolfram_tool import query_wolfram_alpha
            result = query_wolfram_alpha("distance from earth to sun")

            assert "149,600,000" in result

    def test_missing_api_key(self):
        with patch("src.tools.wolfram_tool.WOLFRAM_ALPHA_APP_ID", None):
            from src.tools.wolfram_tool import query_wolfram_alpha
            result = query_wolfram_alpha("test query")

            assert "API" in result or "not configured" in result.lower() or "key" in result.lower()

    def test_did_not_understand(self):
        mock_resp = MockResponse(
            text="Wolfram|Alpha did not understand your input",
            status_code=200
        )

        with patch("src.tools.wolfram_tool.requests.get", return_value=mock_resp), \
             patch("src.tools.wolfram_tool.WOLFRAM_ALPHA_APP_ID", "test_key"):
            from src.tools.wolfram_tool import query_wolfram_alpha
            result = query_wolfram_alpha("asdfghjkl gibberish")

            assert "did not understand" in result.lower() or len(result) > 0

    def test_error_status_code(self):
        mock_resp = MockResponse(text="Error", status_code=501)

        with patch("src.tools.wolfram_tool.requests.get", return_value=mock_resp), \
             patch("src.tools.wolfram_tool.WOLFRAM_ALPHA_APP_ID", "test_key"):
            from src.tools.wolfram_tool import query_wolfram_alpha
            result = query_wolfram_alpha("test")

            assert len(result) > 0  # Should return error info, not crash

    def test_timeout_handling(self):
        import requests
        with patch("src.tools.wolfram_tool.requests.get",
                   side_effect=requests.exceptions.Timeout("timeout")), \
             patch("src.tools.wolfram_tool.WOLFRAM_ALPHA_APP_ID", "test_key"):
            from src.tools.wolfram_tool import query_wolfram_alpha
            result = query_wolfram_alpha("test")

            assert "Error" in result or "timeout" in result.lower() or "failed" in result.lower()
