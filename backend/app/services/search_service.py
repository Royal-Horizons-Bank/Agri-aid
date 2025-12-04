import logging
from typing import Optional, Dict, List
import aiohttp
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class SourceLink:
    """Represents a clickable source link"""

    def __init__(self, title: str, url: str, snippet: str = "", source_type: str = "web"):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source_type = source_type  # 'web', 'official', 'research', 'news'
        self.retrieved_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "type": self.source_type,
            "retrieved_at": self.retrieved_at
        }

    def to_markdown(self) -> str:
        """Convert to clickable markdown link"""
        return f"[{self.title}]({self.url})"

    def to_html(self) -> str:
        """Convert to clickable HTML link"""
        return f'<a href="{self.url}" target="_blank" rel="noopener noreferrer">{self.title}</a>'


# ==================== SERPER API CREDIT SAVING UTILITIES ====================

def _is_weather_query(query: str) -> bool:
    """
    Checks if the query is primarily about weather, forecast, or climate.
    Used to skip Serper API call and save credits.
    """
    weather_keywords = [
        "weather", "forecast", "rain", "sun", "temperature", "typhoon",
        "storm", "climate", "humid", "dry season", "wet season", "cloudy",
        "ulan", "init", "bagyo", "panahon", "pa-asa", "pagasa"
    ]
    normalized_query = query.lower().strip()

    # Check if the query contains any weather-related keyword
    if any(keyword in normalized_query for keyword in weather_keywords):
        return True

    # Simple check for short, direct weather queries
    if len(normalized_query.split()) < 5 and ("weather" in normalized_query or "forecast" in normalized_query):
        return True

    return False


def _is_time_or_date_query(query: str) -> bool:
    """
    Checks if the query is primarily about time, date, or current day.
    Used to skip Serper API call and save credits.
    """
    time_keywords = [
        "time", "date", "today", "now", "current time", "what day is it",
        "araw", "oras", "petsa", "ano oras", "anong araw"
    ]
    normalized_query = query.lower().strip()

    # Check if a time keyword is present and the query is relatively short (to avoid complex queries)
    if any(keyword in normalized_query for keyword in time_keywords) and len(normalized_query.split()) < 7:
        return True

    return False


# ============================================================================


class SearchService:
    """Web search service using Serper API - RESTRICTED TO PHILIPPINES"""

    def __init__(self):
        # Using SERPER_API_KEY from the environment (as indicated in .env)
        # Using the key provided by the user as a fallback
        self.api_key = os.getenv("SERPER_API_KEY", "19acd2ad97533d1f167c8e7b0acfd4c240fab942")
        self.base_url = "https://google.serper.dev/search"
        self.timeout = aiohttp.ClientTimeout(total=10)

        if not self.api_key:
            logger.warning("SERPER_API_KEY not configured. Web searches will use fallback.")

    async def search(self, query: str, num_results: int = 5, force: bool = False) -> Dict:
        """
        Perform web search using Serper API - STRICTLY LOCAL RESULTS

        Args:
            query: Search term
            num_results: Number of results to return
            force: If True, bypasses credit-saving checks (e.g. for explicit weather fallbacks)
        """
        # --- SKIP CHECK FOR WEATHER/TIME/DATE QUERIES TO SAVE CREDITS ---
        if not force:
            if _is_weather_query(query):
                logger.info(f"Skipping Serper API call: Weather query detected for '{query}'.")
                return self._get_fallback_results(query)

            if _is_time_or_date_query(query):
                logger.info(f"Skipping Serper API call: Time/Date query detected for '{query}'.")
                return self._get_fallback_results(query)
        else:
            logger.info(f"Forcing Serper API call for: '{query}'")
        # ---------------------------------------------------------------------

        try:
            if not self.api_key:
                logger.warning("SERPER_API_KEY not configured")
                return self._get_fallback_results(query)

            headers = {
                # Serper API requires the key in the header, not as a parameter
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }

            # FORCE LOCALIZATION: Append 'Philippines' if not present to bias results
            search_query = query
            if "philippines" not in search_query.lower() and "pilipinas" not in search_query.lower():
                search_query = f"{search_query} Philippines"

            # Serper API parameters - STRICTLY PHILIPPINES
            params = {
                "q": search_query,
                "num": num_results,
                "gl": "ph",  # Geolocation: Philippines
                "hl": "en"  # Host Language: English (common for PH gov sites)
            }

            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                # Use GET request with headers
                async with session.get(self.base_url, params=params, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return self._parse_results(data, query)
                    else:
                        logger.error(f"Serper API failed with status {resp.status}: {await resp.text()}")
                        return self._get_fallback_results(query)
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return self._get_fallback_results(query)

    def _parse_results(self, data: Dict, query: str) -> Dict:
        """Parse Serper API results into structured format"""
        results = []
        sources = []

        # Serper API returns organic results under the key 'organic'
        for result in data.get("organic", [])[:5]:
            # Optional: Filter out obviously international domains if needed
            # But gl=ph usually handles ranking well.

            source_obj = SourceLink(
                title=result.get("title", ""),
                url=result.get("link", ""),
                snippet=result.get("snippet", ""),
                source_type="web"
            )
            results.append({
                "title": result.get("title"),
                "url": result.get("link"),
                "snippet": result.get("snippet"),
                # Serper provides a 'source' key that is often the domain name
                "source": result.get("source", "Serper Search"),
                "date": result.get("date", "")
            })
            sources.append(source_obj)

        return {
            "query": query,
            "organic_results": results,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        }

    def _get_fallback_results(self, query: str) -> Dict:
        """Fallback results when API is unavailable or search is skipped."""
        logger.info(f"Using fallback search for: {query}")
        return {
            "query": query,
            "organic_results": [],
            "sources": [],
            "timestamp": datetime.now().isoformat()
        }


# Initialize service
search_service = SearchService()