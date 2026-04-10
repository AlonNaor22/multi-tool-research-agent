"""Shared test fixtures and helpers for the research agent test suite."""

import os
import sys
import json
import pytest
import aiohttp
from unittest.mock import patch, AsyncMock

# Ensure the project root is on the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class MockResponse:
    """Reusable mock for requests.get() responses."""

    def __init__(self, json_data=None, text="", status_code=200, content=b"",
                 headers=None):
        self._json_data = json_data
        self.text = text
        self.status_code = status_code
        self.content = content
        self.headers = headers or {"Content-Type": "text/html"}

    def json(self):
        if self._json_data is not None:
            return self._json_data
        raise ValueError("No JSON data")

    def raise_for_status(self):
        if self.status_code >= 400:
            from requests.exceptions import HTTPError
            raise HTTPError(f"{self.status_code} Error")


class AsyncMockResponse:
    """Reusable mock for aiohttp responses."""

    def __init__(self, json_data=None, text="", status=200, content=b"",
                 headers=None):
        self._json_data = json_data
        self._text = text
        self.status = status
        self._content = content
        self.headers = headers or {"Content-Type": "text/html"}

    async def json(self):
        if self._json_data is not None:
            return self._json_data
        raise ValueError("No JSON data")

    async def text(self):
        return self._text

    async def read(self):
        return self._content

    def raise_for_status(self):
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                None, None, status=self.status, message=f"{self.status} Error"
            )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.fixture(autouse=True, scope="session")
def _fast_retries():
    """Zero-out retry delays so error-handling tests don't waste wall-clock
    time on backoff sleeps.

    Patches only ``src.utils._retry_sleep`` — the wrapper used exclusively
    by the retry decorator. ``asyncio.sleep`` itself is unaffected, so
    timeout tests and other sleep-dependent code work normally.
    """
    with patch("src.utils._retry_sleep", new_callable=AsyncMock):
        yield


@pytest.fixture
def mock_response():
    """Factory fixture for creating MockResponse objects."""
    def _make(**kwargs):
        return MockResponse(**kwargs)
    return _make


@pytest.fixture
def async_mock_response():
    """Factory fixture for creating AsyncMockResponse objects."""
    def _make(**kwargs):
        return AsyncMockResponse(**kwargs)
    return _make


@pytest.fixture
def weather_api_response():
    """Sample weather API response data."""
    return {
        "weather": [{"description": "clear sky"}],
        "main": {"temp": 22, "feels_like": 21, "humidity": 45},
        "wind": {"speed": 3.5},
        "name": "London",
        "sys": {"country": "GB"},
    }


@pytest.fixture
def forecast_api_response():
    """Sample forecast API response data."""
    return {
        "city": {"name": "London", "country": "GB"},
        "list": [
            {
                "dt_txt": "2026-03-23 12:00:00",
                "weather": [{"description": "sunny"}],
                "main": {"temp": 20, "temp_min": 15, "temp_max": 25},
            },
            {
                "dt_txt": "2026-03-24 12:00:00",
                "weather": [{"description": "cloudy"}],
                "main": {"temp": 18, "temp_min": 13, "temp_max": 22},
            },
        ],
    }


@pytest.fixture
def search_results():
    """Sample DuckDuckGo search results."""
    return [
        {
            "title": "Test Result 1",
            "href": "https://example.com/1",
            "body": "This is the first result snippet.",
        },
        {
            "title": "Test Result 2",
            "href": "https://example.com/2",
            "body": "This is the second result snippet.",
        },
    ]
