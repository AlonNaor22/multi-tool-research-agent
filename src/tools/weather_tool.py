"""Weather tool for the research agent.

Uses OpenWeatherMap API to get current weather and forecasts.
Requires an API key from https://openweathermap.org/api

Features:
- Current weather conditions
- 5-day forecast (3-hour intervals)
- Support for city names or coordinates
- Configurable units (metric/imperial)
"""

import os
import json
import requests
from langchain_core.tools import Tool
from src.utils import retry_on_error


# API endpoints
CURRENT_WEATHER_URL = "http://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "http://api.openweathermap.org/data/2.5/forecast"


def _format_current_weather(data: dict, units: str) -> str:
    """Format current weather data into readable string."""
    temp_unit = "°C" if units == "metric" else "°F"
    speed_unit = "m/s" if units == "metric" else "mph"

    weather_desc = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]
    wind_speed = data["wind"]["speed"]
    city_name = data["name"]
    country = data["sys"]["country"]

    return (
        f"**Current weather in {city_name}, {country}:**\n"
        f"Conditions: {weather_desc.capitalize()}\n"
        f"Temperature: {temp}{temp_unit} (feels like {feels_like}{temp_unit})\n"
        f"Humidity: {humidity}%\n"
        f"Wind speed: {wind_speed} {speed_unit}"
    )


def _format_forecast(data: dict, units: str, days: int = 3) -> str:
    """Format forecast data into readable string."""
    temp_unit = "°C" if units == "metric" else "°F"

    city_name = data["city"]["name"]
    country = data["city"]["country"]

    # Group forecasts by day (take one reading per day around noon)
    daily_forecasts = {}
    for item in data["list"]:
        date = item["dt_txt"].split(" ")[0]
        hour = int(item["dt_txt"].split(" ")[1].split(":")[0])

        # Prefer readings around noon (12:00)
        if date not in daily_forecasts or abs(hour - 12) < abs(int(daily_forecasts[date]["dt_txt"].split(" ")[1].split(":")[0]) - 12):
            daily_forecasts[date] = item

    # Format output
    result_parts = [f"**{days}-day forecast for {city_name}, {country}:**\n"]

    count = 0
    for date, item in sorted(daily_forecasts.items()):
        if count >= days:
            break

        weather_desc = item["weather"][0]["description"]
        temp = item["main"]["temp"]
        temp_min = item["main"]["temp_min"]
        temp_max = item["main"]["temp_max"]

        result_parts.append(
            f"{date}: {weather_desc.capitalize()}, "
            f"{temp_min:.0f}-{temp_max:.0f}{temp_unit} (avg {temp:.0f}{temp_unit})"
        )
        count += 1

    return "\n".join(result_parts)


@retry_on_error(
    max_retries=2,
    delay=1.0,
    exceptions=(requests.exceptions.Timeout, requests.exceptions.ConnectionError)
)
def get_weather(query: str) -> str:
    """
    Get weather information for a location.

    Input can be:
    - Simple city name: "London"
    - JSON with options: {"location": "London", "units": "imperial", "forecast": true}
    - Coordinates: {"lat": 51.5, "lon": -0.1}

    Args:
        query: City name string or JSON with options

    Returns:
        Weather information as formatted string, or error message.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return (
            "Weather API key not configured. "
            "Add OPENWEATHER_API_KEY to your .env file. "
            "Get a free key at: https://openweathermap.org/api"
        )

    # Parse input
    location = None
    lat = None
    lon = None
    units = "metric"  # Default to Celsius
    forecast = False
    forecast_days = 3

    try:
        if query.strip().startswith("{"):
            options = json.loads(query)
            location = options.get("location")
            lat = options.get("lat")
            lon = options.get("lon")
            units = options.get("units", "metric")
            forecast = options.get("forecast", False)
            forecast_days = min(options.get("days", 3), 5)  # Max 5 days
        else:
            location = query
    except json.JSONDecodeError:
        location = query

    # Validate units
    if units not in ("metric", "imperial"):
        units = "metric"

    # Build API parameters
    params = {
        "appid": api_key,
        "units": units
    }

    if lat is not None and lon is not None:
        params["lat"] = lat
        params["lon"] = lon
    elif location:
        params["q"] = location
    else:
        return "Error: Provide a location (city name) or coordinates (lat/lon)."

    try:
        if forecast:
            # Get forecast
            response = requests.get(FORECAST_URL, params=params, timeout=10)
            data = response.json()

            if response.status_code == 200:
                return _format_forecast(data, units, forecast_days)
            elif response.status_code == 404:
                return f"Location not found. Try a different city name."
            else:
                return f"Weather API error: {data.get('message', 'Unknown error')}"
        else:
            # Get current weather
            response = requests.get(CURRENT_WEATHER_URL, params=params, timeout=10)
            data = response.json()

            if response.status_code == 200:
                return _format_current_weather(data, units)
            elif response.status_code == 404:
                return f"City not found. Try a different city name."
            else:
                return f"Weather API error: {data.get('message', 'Unknown error')}"

    except requests.exceptions.Timeout:
        return "Weather request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Weather request failed: {str(e)}"
    except KeyError as e:
        return f"Unexpected weather data format: missing {str(e)}"


# Create the LangChain Tool wrapper
weather_tool = Tool(
    name="weather",
    func=get_weather,
    description=(
        "Get current weather or forecast for a location. "
        "\n\nSIMPLE USAGE: Just provide a city name: 'London', 'New York', 'Tokyo'"
        "\n\nADVANCED USAGE: Provide JSON with options:"
        '\n{"location": "Paris", "units": "imperial", "forecast": true, "days": 5}'
        "\n\nOPTIONS:"
        "\n- location: City name"
        "\n- lat/lon: Coordinates (instead of city name)"
        "\n- units: 'metric' (Celsius, default) or 'imperial' (Fahrenheit)"
        "\n- forecast: true for 5-day forecast, false for current weather (default)"
        "\n- days: Number of forecast days (1-5, default 3)"
        "\n\nEXAMPLES:"
        "\n- 'London'"
        '\n- {"location": "Tokyo", "forecast": true}'
        '\n- {"lat": 40.7, "lon": -74.0, "units": "imperial"}'
    )
)
