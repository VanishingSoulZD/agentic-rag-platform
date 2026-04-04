"""Mock weather API wrapper tool."""

from __future__ import annotations

MOCK_WEATHER_DATA = {
    "beijing": "Cloudy, 18°C",
    "shanghai": "Rainy, 20°C",
    "taipei": "Sunny, 25°C",
    "hangzhou": "Clear, 24°C",
}


def get_weather(city: str) -> str:
    """Return mock weather result for a city name."""
    key = city.strip().lower()
    if not key:
        raise ValueError("City must not be empty.")

    forecast = MOCK_WEATHER_DATA.get(key)
    if forecast is None:
        return f"Weather for {city} is unavailable in mock API"
    return f"{city}: {forecast}"
