import logging
from typing import Dict, Optional, Any
import aiohttp
import os
from datetime import datetime
# Import search_service to enable PAGASA scraping via Serper
from app.services.search_service import search_service
# Import settings to ensure .env variables are loaded correctly
from app.config import settings

logger = logging.getLogger(__name__)


class WeatherService:
    """
    Real-time weather data service.
    Primary: WeatherAPI.com (via API Key)
    Secondary: PAGASA (via Serper Web Scraping)
    """

    def __init__(self):
        # FIX: Use settings.WEATHERAPI_KEY instead of os.getenv to ensure .env is loaded
        self.api_key = settings.WEATHERAPI_KEY
        self.base_url = "http://api.weatherapi.com/v1"
        self.timeout = aiohttp.ClientTimeout(total=10)

        if not self.api_key:
            logger.warning("WEATHERAPI_KEY not found in settings. Weather API calls will fail.")

    async def get_weather(self, location: str) -> Dict:
        """
        Get real-time weather data for a location.
        Tries WeatherAPI first, then falls back to PAGASA via Search.
        """
        try:
            # 1. Try WeatherAPI.com (Primary)
            if self.api_key:
                weather_data = await self._get_weatherapi(location, "current.json")
                if weather_data:
                    return weather_data
            else:
                logger.warning("WeatherAPI Key is missing.")

            # 2. Try PAGASA via Serper Search (Fallback/Scraping)
            logger.info(f"WeatherAPI failed or key missing. Attempting PAGASA search for {location}...")
            pagasa_data = await self._get_pagasa_via_search(location)
            if pagasa_data:
                return pagasa_data

            # 3. Default Fallback
            return self._get_default_weather(location)

        except Exception as e:
            logger.error(f"Weather service error: {str(e)}")
            return self._get_default_weather(location)

    async def _get_weatherapi(self, location: str, endpoint: str) -> Optional[Dict]:
        """Get weather from WeatherAPI.com"""
        try:
            if not self.api_key:
                return None

            url = f"{self.base_url}/{endpoint}"
            params = {
                "key": self.api_key,
                "q": location,
                "days": 7 if endpoint == "forecast.json" else 1,
                "aqi": "no"
            }

            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if endpoint == "current.json":
                            return self._format_weatherapi_current(data, location)
                        elif endpoint == "forecast.json":
                            return self._format_weatherapi_forecast(data, location)
                    else:
                        logger.warning(f"WeatherAPI error {resp.status}: {await resp.text()}")
            return None
        except Exception as e:
            logger.warning(f"WeatherAPI exception: {str(e)}")
            return None

    async def _get_pagasa_via_search(self, location: str) -> Optional[Dict]:
        """
        'Scrape' PAGASA weather info by searching via Serper.
        This finds the latest forecast link and snippets.
        """
        try:
            query = f"PAGASA weather forecast {location} today"
            # Use force=True to bypass the credit-saving check in search_service
            search_result = await search_service.search(query, num_results=3, force=True)

            if not search_result or not search_result.get("organic_results"):
                return None

            # Extract best result
            top_result = search_result["organic_results"][0]
            snippet = top_result.get("snippet", "No details available.")
            title = top_result.get("title", "PAGASA Update")
            link = top_result.get("url", "https://bagong.pagasa.dost.gov.ph/")

            # Attempt to extract basic info from snippet (naive extraction)
            condition = "Unknown"
            if "rain" in snippet.lower() or "shower" in snippet.lower():
                condition = "Rainy/Cloudy"
            elif "sunny" in snippet.lower() or "fair" in snippet.lower():
                condition = "Sunny/Fair"
            elif "storm" in snippet.lower() or "typhoon" in snippet.lower():
                condition = "Stormy"

            return {
                "location": location,
                "temperature": "See link",  # Hard to extract exact num from snippet reliably without NLP
                "humidity": "N/A",
                "condition": condition,
                "wind_speed": "N/A",
                "rainfall": "Check forecast",
                "note": f"Data retrieved via search: {snippet[:100]}...",
                "source_url": link,
                "source_title": f"PAGASA via Search: {title}",
                "last_updated": datetime.now().isoformat(),
                "sources": [
                    {
                        "title": title,
                        "url": link,
                        "type": "official"
                    }
                ]
            }

        except Exception as e:
            logger.error(f"PAGASA search error: {str(e)}")
            return None

    def _format_weatherapi_current(self, data: Dict[str, Any], location: str) -> Dict:
        """Format WeatherAPI current response"""
        current = data.get("current", {})
        location_data = data.get("location", {})

        return {
            "location": location_data.get("name", location),
            "region": location_data.get("region"),
            "temperature": current.get("temp_c"),
            "feels_like": current.get("feelslike_c"),
            "humidity": current.get("humidity"),
            "pressure": current.get("pressure_mb"),
            "condition": current.get("condition", {}).get("text"),
            "wind_speed": current.get("wind_kph"),
            "wind_direction": current.get("wind_dir"),
            "rainfall": current.get("precip_mm"),
            "uv_index": current.get("uv"),
            "is_day": current.get("is_day") == 1,
            "source_url": "https://www.weatherapi.com/",
            "source_title": "WeatherAPI.com",
            "last_updated": location_data.get("localtime", datetime.now().isoformat()),
            "sources": [
                {
                    "title": "WeatherAPI.com",
                    "url": "https://www.weatherapi.com/",
                    "type": "weather"
                }
            ]
        }

    def _format_weatherapi_forecast(self, data: Dict[str, Any], location: str) -> Dict:
        """Format WeatherAPI forecast response"""
        location_data = data.get("location", {})
        forecast_days = data.get("forecast", {}).get("forecastday", [])

        forecasts = []
        for day in forecast_days:
            date = day.get("date")
            day_data = day.get("day", {})
            astro = day.get("astro", {})

            forecasts.append({
                "date": date,
                "max_temp": day_data.get("maxtemp_c"),
                "min_temp": day_data.get("mintemp_c"),
                "avg_temp": day_data.get("avgtemp_c"),
                "condition": day_data.get("condition", {}).get("text"),
                "total_rainfall_mm": day_data.get("totalprecip_mm"),
                "avg_humidity": day_data.get("avghumidity"),
                "sunrise": astro.get("sunrise"),
                "sunset": astro.get("sunset")
            })

        return {
            "location": location_data.get("name", location),
            "forecast": forecasts,
            "source_url": "https://www.weatherapi.com/",
            "source_title": "WeatherAPI.com",
            "last_updated": datetime.now().isoformat(),
            "sources": [
                {
                    "title": "WeatherAPI.com Forecast",
                    "url": "https://www.weatherapi.com/",
                    "type": "weather"
                }
            ]
        }

    def _get_default_weather(self, location: str) -> Dict:
        """Default weather when APIs are unavailable"""
        return {
            "location": location,
            "temperature": None,
            "condition": "Data Unavailable",
            "note": "Real-time weather services are currently unreachable.",
            "source_url": "https://www.pagasa.dost.gov.ph",
            "source_title": "PAGASA (Official Website)",
            "last_updated": datetime.now().isoformat(),
            "sources": [
                {
                    "title": "PAGASA Official Website",
                    "url": "https://www.pagasa.dost.gov.ph",
                    "type": "official"
                }
            ]
        }

    async def get_forecast(self, location: str, days: int = 7) -> Dict:
        """Get weather forecast"""
        if self.api_key:
            return await self._get_weatherapi(location, "forecast.json") or {"error": "Forecast unavailable"}
        return {"error": "Forecast unavailable"}


# Initialize service
weather_service = WeatherService()