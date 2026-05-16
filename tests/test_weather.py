"""Tests for src/tools/weather_tool.py -- weather data retrieval."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tests.conftest import AsyncMockResponse
from src.tools.weather_tool import get_weather


class TestCurrentWeather:
    """Test current weather retrieval."""

    async def test_formats_weather_correctly(self, weather_api_response):
        mock_resp = AsyncMockResponse(json_data=weather_api_response, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.tools.weather_tool.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session), \
             patch("src.tools.weather_tool.os.getenv", return_value="test_key"):
            result = await get_weather("London")

            assert "London" in result
            assert "GB" in result
            assert "22" in result
            assert "clear sky" in result.lower()

    async def test_city_not_found(self):
        mock_resp = AsyncMockResponse(
            json_data={"message": "city not found"},
            status=404
        )
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.tools.weather_tool.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session), \
             patch("src.tools.weather_tool.os.getenv", return_value="test_key"):
            result = await get_weather("Nonexistentville")

            assert "not found" in result.lower()

    async def test_no_api_key(self):
        with patch("src.tools.weather_tool.os.getenv", return_value=None):
            result = await get_weather("London")

            assert "API key" in result or "not configured" in result

    async def test_imperial_units(self, weather_api_response):
        weather_api_response["main"]["temp"] = 72
        weather_api_response["main"]["feels_like"] = 70
        mock_resp = AsyncMockResponse(json_data=weather_api_response, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.tools.weather_tool.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session), \
             patch("src.tools.weather_tool.os.getenv", return_value="test_key"):
            result = await get_weather(location="London", units="imperial")

            assert "72" in result


class TestForecast:
    """Test forecast retrieval."""

    async def test_forecast_formatting(self, forecast_api_response):
        mock_resp = AsyncMockResponse(json_data=forecast_api_response, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.tools.weather_tool.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session), \
             patch("src.tools.weather_tool.os.getenv", return_value="test_key"):
            result = await get_weather(location="London", forecast=True)

            assert "London" in result
            assert "forecast" in result.lower()


class TestWeatherEdgeCases:
    """Test edge cases."""

    async def test_empty_location(self):
        with patch("src.tools.weather_tool.os.getenv", return_value="test_key"):
            result = await get_weather(units="metric")

            assert "Error" in result or "location" in result.lower()

    async def test_coordinates_input(self, weather_api_response):
        mock_resp = AsyncMockResponse(json_data=weather_api_response, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.tools.weather_tool.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session), \
             patch("src.tools.weather_tool.os.getenv", return_value="test_key"):
            result = await get_weather(lat=51.5, lon=-0.1)

            assert "London" in result
