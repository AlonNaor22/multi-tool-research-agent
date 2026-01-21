"""Weather tool for the research agent.

Uses OpenWeatherMap API to get current weather data for any city.
Requires an API key from https://openweathermap.org/api

This gives the agent access to real-time weather information.
"""

import os
import requests
from langchain_core.tools import Tool
from src.utils import retry_on_error


@retry_on_error(
    max_retries=2,
    delay=1.0,
    exceptions=(requests.exceptions.Timeout, requests.exceptions.ConnectionError)
)
def get_weather(location: str) -> str:
    """
    Get current weather for a location using OpenWeatherMap API.

    Args:
        location: City name (e.g., "London", "New York", "Tokyo")

    Returns:
        Weather information as a formatted string, or error message.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    # Check if API key is configured
    if not api_key:
        return (
            "Weather API key not configured. "
            "Add OPENWEATHER_API_KEY to your .env file. "
            "Get a free key at: https://openweathermap.org/api"
        )

    # OpenWeatherMap API endpoint
    base_url = "http://api.openweathermap.org/data/2.5/weather"

    # Parameters for the API request
    params = {
        "q": location,        # City name
        "appid": api_key,     # API key
        "units": "metric"     # Use Celsius (use "imperial" for Fahrenheit)
    }

    try:
        # Make the HTTP request to the API
        response = requests.get(base_url, params=params, timeout=10)
        data = response.json()

        # Check if the request was successful
        if response.status_code == 200:
            # Extract weather data from the response
            weather_desc = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            feels_like = data["main"]["feels_like"]
            humidity = data["main"]["humidity"]
            wind_speed = data["wind"]["speed"]
            city_name = data["name"]
            country = data["sys"]["country"]

            # Format the response nicely
            return (
                f"Weather in {city_name}, {country}: {weather_desc.capitalize()}. "
                f"Temperature: {temp}°C (feels like {feels_like}°C). "
                f"Humidity: {humidity}%. "
                f"Wind speed: {wind_speed} m/s."
            )
        elif response.status_code == 404:
            return f"City '{location}' not found. Try a different city name."
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
        "Get current weather information for a city. Use this when the user asks "
        "about weather, temperature, or climate conditions in a specific location. "
        "Input should be a city name (e.g., 'London', 'New York', 'Tokyo', 'Paris'). "
        "Returns temperature, humidity, wind speed, and weather description."
    )
)
