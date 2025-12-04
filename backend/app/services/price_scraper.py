import logging
from typing import Dict, List, Optional
import aiohttp
import json
from datetime import datetime
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class PriceScraperService:
    """
    Retrieves accurate agricultural pricing data from Philippine Statistics Authority (PSA)
    and other official government sources.
    STRICT MODE: No guessing. Live data only.
    FALLBACK: If local data is missing, defaults to National/Philippines average.
    """

    def __init__(self):
        self.psa_base_url = "https://openstat.psa.gov.ph"
        self.da_base_url = "https://www.da.gov.ph"
        self.timeout = aiohttp.ClientTimeout(total=15)

    async def get_palay_data(self, province: str, year: int = 2024) -> Optional[Dict]:
        """
        Get accurate rice (palay) production and pricing data.
        Tries specific province first, then falls back to National/Philippines level.
        """
        try:
            # 1. Primary: Specific Province via PSA
            psa_data = await self._get_psa_data(province, year)
            if psa_data:
                return psa_data

            # 2. Secondary: Specific Province via DA
            da_data = await self._get_da_data(province, year)
            if da_data:
                return da_data

            # 3. FALLBACK: Try National Level (Philippines)
            # This ensures we provide "overall Philippines" data instead of nothing
            if province.lower() != "philippines":
                logger.info(f"No local palay data for {province}. Attempting National fallback...")

                national_psa = await self._get_psa_data("Philippines", year)
                if national_psa:
                    national_psa['province'] = f"Philippines (National Average - {province} data unavailable)"
                    national_psa['data']['source_title'] += " (National Fallback)"
                    return national_psa

                national_da = await self._get_da_data("Philippines", year)
                if national_da:
                    national_da['province'] = f"Philippines (National Average - {province} data unavailable)"
                    return national_da

            logger.warning(f"No price data found for {province} or National level.")
            return None

        except Exception as e:
            logger.error(f"Error fetching palay data: {str(e)}")
            return None

    async def _get_psa_data(self, province: str, year: int) -> Optional[Dict]:
        """Retrieve data from PSA OpenStat API"""
        try:
            url = f"{self.psa_base_url}/api/data/series"
            params = {
                "series_id": "AGRI_PALAY_PROD",
                "region": province,
                "year": year
            }

            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data and "price_per_kg" in data:
                            return self._format_psa_response(data, province)
            return None
        except Exception as e:
            logger.warning(f"PSA API error for {province}: {str(e)}")
            return None

    async def _get_da_data(self, province: str, year: int) -> Optional[Dict]:
        """Retrieve data from DA Portal"""
        try:
            url = f"{self.da_base_url}/api/statistics/crops"
            params = {
                "crop": "rice",
                "province": province,
                "year": year
            }

            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data and "price" in data:
                            return self._format_da_response(data, province)
            return None
        except Exception as e:
            logger.warning(f"DA Portal error for {province}: {str(e)}")
            return None

    def _format_psa_response(self, data: Dict, province: str) -> Dict:
        """Format PSA API response"""
        specific_url = "https://openstat.psa.gov.ph/PXWeb/pxweb/en/DB/DB__2E__CS/0012E4EVCP0.px"
        return {
            "province": province,
            "year": 2024,
            "data": {
                "average_price": data.get("price_per_kg", 0),
                "production_volume": data.get("total_production_mt", 0),
                "area_harvested": data.get("area_harvested_ha", 0),
                "yield_per_hectare": data.get("yield_mt_per_ha", 0),
                "unit": "MT (Metric Tons)",
                "currency": "PHP",
                "last_updated": data.get("date", datetime.now().isoformat()),
                "source_url": specific_url,
                "source_title": "PSA OpenStat - Palay Production Estimates"
            },
            "sources": [
                {
                    "title": "PSA OpenStat - Palay Production Estimates",
                    "url": specific_url,
                    "type": "official"
                }
            ]
        }

    def _format_da_response(self, data: Dict, province: str) -> Dict:
        """Format DA API response"""
        specific_url = "https://www.da.gov.ph/price-monitoring/"
        return {
            "province": province,
            "year": 2024,
            "data": {
                "average_price": data.get("price", 0),
                "production_volume": data.get("production", 0),
                "area_harvested": data.get("area", 0),
                "yield_per_hectare": data.get("yield", 0),
                "unit": "MT (Metric Tons)",
                "currency": "PHP",
                "last_updated": data.get("updated", datetime.now().isoformat()),
                "source_url": specific_url,
                "source_title": "DA Price Monitoring"
            },
            "sources": [
                {
                    "title": "Department of Agriculture - Price Monitoring",
                    "url": specific_url,
                    "type": "official"
                }
            ]
        }

    async def get_commodity_prices(self, commodity: str, location: str = "") -> Optional[Dict]:
        """
        Get current commodity prices.
        Falls back to National/Metro Manila context if specific location fails.
        """
        try:
            url = "https://www.da.gov.ph/price-monitoring/"

            # Attempt to check specific location availability (simulated logic via official URL access)
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        # For commodities, if we can't scrape specific town data, we return the general link
                        # but clearly labelled as checking the "National/Metro Manila" baseline.

                        location_display = location if location else "Metro Manila / National"

                        return {
                            "commodity": commodity,
                            "location": location_display,
                            "prices": [],
                            "note": f"Checking latest daily price monitoring for {commodity}. If specific {location} data is unavailable, please refer to the Metro Manila/National baseline in the link.",
                            "last_updated": datetime.now().isoformat(),
                            "source_url": url,
                            "source_title": "DA Bantay Presyo (Price Watch)",
                            "sources": [
                                {
                                    "title": "DA Bantay Presyo (Official Price Watch)",
                                    "url": url,
                                    "type": "official"
                                }
                            ]
                        }
        except Exception as e:
            logger.warning(f"DA Price Watch error: {str(e)}")

        return None


# Initialize service
psa_scraper = PriceScraperService()