"""Weather tool — uses OpenWeatherMap API for current weather and forecasts."""

import os
import asyncio
from typing import Type, Literal, Optional
import aiohttp
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from src.utils import async_retry_on_error, parse_tool_input, get_aiohttp_session, safe_tool_call


# API endpoints
CURRENT_WEATHER_URL = "http://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "http://api.openweathermap.org/data/2.5/forecast"


def _format_current_weather(data: dict, units: str) -> str:
    """Format current weather API response into a readable string."""
    temp_unit = "\u00b0C" if units == "metric" else "\u00b0F"
    speed_unit = "m/s" if units == "metric" else "mph"

    weather_list = data.get("weather", [{}])
    weather_desc = weather_list[0].get("description", "Unknown") if weather_list else "Unknown"
    main = data.get("main", {})
    temp = main.get("temp", "N/A")
    feels_like = main.get("feels_like", "N/A")
    humidity = main.get("humidity", "N/A")
    wind_speed = data.get("wind", {}).get("speed", "N/A")
    city_name = data.get("name", "Unknown")
    country = data.get("sys", {}).get("country", "")

    return (
        f"**Current weather in {city_name}, {country}:**\n"
        f"Conditions: {weather_desc.capitalize()}\n"
        f"Temperature: {temp}{temp_unit} (feels like {feels_like}{temp_unit})\n"
        f"Humidity: {humidity}%\n"
        f"Wind speed: {wind_speed} {speed_unit}"
    )


def _extract_hour(dt_txt: str) -> int:
    """Extract the hour from an OpenWeatherMap dt_txt string."""
    try:
        return int(dt_txt.split(" ")[1].split(":")[0])
    except (IndexError, ValueError):
        return 0


def _is_closer_to_noon(candidate_hour: int, current_best_hour: int) -> bool:
    """Return True if candidate_hour is closer to 12:00 than current_best_hour."""
    return abs(candidate_hour - 12) < abs(current_best_hour - 12)


def _format_forecast(data: dict, units: str, days: int = 3) -> str:
    """Format forecast API response into a readable multi-day summary."""
    temp_unit = "\u00b0C" if units == "metric" else "\u00b0F"

    city = data.get("city", {})
    city_name = city.get("name", "Unknown")
    country = city.get("country", "")

    daily_forecasts = {}
    for item in data.get("list", []):
        dt_txt = item.get("dt_txt", "")
        date = dt_txt.split(" ")[0] if " " in dt_txt else dt_txt
        hour = _extract_hour(dt_txt)

        if date not in daily_forecasts or _is_closer_to_noon(hour, _extract_hour(daily_forecasts[date].get("dt_txt", ""))):
            daily_forecasts[date] = item

    result_parts = [f"**{days}-day forecast for {city_name}, {country}:**\n"]

    count = 0
    for date, item in sorted(daily_forecasts.items()):
        if count >= days:
            break

        weather_list = item.get("weather", [{}])
        weather_desc = weather_list[0].get("description", "Unknown") if weather_list else "Unknown"
        main = item.get("main", {})
        temp = main.get("temp", 0)
        temp_min = main.get("temp_min", 0)
        temp_max = main.get("temp_max", 0)

        result_parts.append(
            f"{date}: {weather_desc.capitalize()}, "
            f"{temp_min:.0f}-{temp_max:.0f}{temp_unit} (avg {temp:.0f}{temp_unit})"
        )
        count += 1

    return "\n".join(result_parts)


@safe_tool_call("getting weather")
@async_retry_on_error(
    max_retries=2,
    delay=1.0,
    exceptions=(aiohttp.ClientError, asyncio.TimeoutError)
)
async def _get_weather(location: str = "", lat: float = None, lon: float = None,
                       units: str = "metric", forecast: bool = False, days: int = 3) -> str:
    """Core async weather logic."""
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return (
            "Weather API key not configured. "
            "Add OPENWEATHER_API_KEY to your .env file. "
            "Get a free key at: https://openweathermap.org/api"
        )

    if units not in ("metric", "imperial"):
        units = "metric"

    forecast_days = min(days, 5)

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

    session = await get_aiohttp_session()
    url = FORECAST_URL if forecast else CURRENT_WEATHER_URL
    async with session.get(
        url, params=params,
        timeout=aiohttp.ClientTimeout(total=10),
    ) as resp:
        data = await resp.json()
        if resp.status != 200:
            return f"Weather API error: {data.get('message', 'Unknown error')}"

        if forecast:
            return _format_forecast(data, units, forecast_days)
        return _format_current_weather(data, units)


# ---------------------------------------------------------------------------
# BaseTool subclass
# ---------------------------------------------------------------------------

class WeatherInput(BaseModel):
    location: str = Field(default="", description="City name (e.g. 'London', 'New York')")
    lat: Optional[float] = Field(default=None, description="Latitude for coordinate-based lookup")
    lon: Optional[float] = Field(default=None, description="Longitude for coordinate-based lookup")
    units: Literal["metric", "imperial"] = Field(default="metric", description="Temperature units")
    forecast: bool = Field(default=False, description="If true, return multi-day forecast")
    days: int = Field(default=3, ge=1, le=5, description="Number of forecast days (1-5)")


class WeatherTool(BaseTool):
    name: str = "weather"
    description: str = (
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
    args_schema: Type[BaseModel] = WeatherInput

    async def _arun(self, location: str = "", lat: float = None, lon: float = None,
                    units: str = "metric", forecast: bool = False, days: int = 3) -> str:
        return await _get_weather(location, lat, lon, units, forecast, days)

    def _run(self, **kwargs) -> str:
        import asyncio
        return asyncio.run(self._arun(**kwargs))


weather_tool = WeatherTool()


async def get_weather(query: str) -> str:
    """Parse a location string/JSON and fetch weather. Used by tests and CLI."""
    location_str, opts = parse_tool_input(query, {
        "units": "metric", "forecast": False, "days": 3,
    })
    if location_str and location_str.strip().startswith("{"):
        location_str = ""
    return await _get_weather(
        location=opts.get("location", location_str or ""),
        lat=opts.get("lat"), lon=opts.get("lon"),
        units=opts.get("units", "metric"),
        forecast=opts.get("forecast", False),
        days=min(opts.get("days", 3), 5),
    )
