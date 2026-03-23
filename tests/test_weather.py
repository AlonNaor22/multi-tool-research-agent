"""Tests for src/tools/weather_tool.py — weather data retrieval."""

import pytest
from unittest.mock import patch, MagicMock
from tests.conftest import MockResponse
from src.tools.weather_tool import get_weather


class TestCurrentWeather:
    """Test current weather retrieval."""

    def test_formats_weather_correctly(self, weather_api_response):
        mock_resp = MockResponse(json_data=weather_api_response, status_code=200)

        with patch("src.tools.weather_tool.requests.get", return_value=mock_resp), \
             patch("src.tools.weather_tool.os.getenv", return_value="test_key"):
            result = get_weather("London")

            assert "London" in result
            assert "GB" in result
            assert "22" in result
            assert "clear sky" in result.lower()

    def test_city_not_found(self):
        mock_resp = MockResponse(
            json_data={"message": "city not found"},
            status_code=404
        )

        with patch("src.tools.weather_tool.requests.get", return_value=mock_resp), \
             patch("src.tools.weather_tool.os.getenv", return_value="test_key"):
            result = get_weather("Nonexistentville")

            assert "not found" in result.lower()

    def test_no_api_key(self):
        with patch("src.tools.weather_tool.os.getenv", return_value=None):
            result = get_weather("London")

            assert "API key" in result or "not configured" in result

    def test_imperial_units(self, weather_api_response):
        weather_api_response["main"]["temp"] = 72
        weather_api_response["main"]["feels_like"] = 70
        mock_resp = MockResponse(json_data=weather_api_response, status_code=200)

        with patch("src.tools.weather_tool.requests.get", return_value=mock_resp), \
             patch("src.tools.weather_tool.os.getenv", return_value="test_key"):
            result = get_weather('{"location": "London", "units": "imperial"}')

            assert "72" in result


class TestForecast:
    """Test forecast retrieval."""

    def test_forecast_formatting(self, forecast_api_response):
        mock_resp = MockResponse(json_data=forecast_api_response, status_code=200)

        with patch("src.tools.weather_tool.requests.get", return_value=mock_resp), \
             patch("src.tools.weather_tool.os.getenv", return_value="test_key"):
            result = get_weather('{"location": "London", "forecast": true}')

            assert "London" in result
            assert "forecast" in result.lower()


class TestWeatherEdgeCases:
    """Test edge cases."""

    def test_empty_location(self):
        with patch("src.tools.weather_tool.os.getenv", return_value="test_key"):
            result = get_weather('{"units": "metric"}')

            assert "Error" in result or "location" in result.lower()

    def test_coordinates_input(self, weather_api_response):
        mock_resp = MockResponse(json_data=weather_api_response, status_code=200)

        with patch("src.tools.weather_tool.requests.get", return_value=mock_resp), \
             patch("src.tools.weather_tool.os.getenv", return_value="test_key"):
            result = get_weather('{"lat": 51.5, "lon": -0.1}')

            assert "London" in result
