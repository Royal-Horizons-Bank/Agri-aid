from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    """
    Application settings with validation.
    Reads from environment variables (case-insensitive) or .env file.
    """

    # =====================
    # APPLICATION SETTINGS
    # =====================
    APP_NAME: str = "Agri-Aid"
    DEBUG: bool = True
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ENVIRONMENT: str = "development"

    # =====================
    # OLLAMA CONFIGURATION
    # =====================
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"
    OLLAMA_VISION_MODEL: str = "llava"  # New: For image analysis
    OLLAMA_TEMPERATURE: float = 0.7
    OLLAMA_KEEP_ALIVE: str = "24h"

    # =====================
    # PSA CONFIGURATION
    # =====================
    PSA_BASE_URL: str = "https://openstat.psa.gov.ph"
    PSA_API_ENDPOINT: str = "https://openstat.psa.gov.ph/api/3/action"
    PSA_CACHE_TTL: int = 3600
    PSA_SCRAPE_TIMEOUT: int = 15

    # =====================
    # WEATHER CONFIGURATION
    # =====================
    # Primary: PAGASA via Search (No key required, logic handled in service)
    # Legacy keys kept for backward compatibility if needed
    WEATHERAPI_KEY: Optional[str] = None
    OPENWEATHER_API_KEY: Optional[str] = None
    WEATHER_API_TYPE: str = "pagasa_search"

    # =====================
    # SERPER WEB SEARCH (Required for PH Localization)
    # =====================
    SERPER_API_KEY: str = ""

    # =====================
    # VISION APIs (Optional/Fallback)
    # =====================
    GOOGLE_VISION_API_KEY: Optional[str] = None
    CLAUDE_API_KEY: Optional[str] = None

    # =====================
    # WEB SCRAPING SETTINGS
    # =====================
    SCRAPE_USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    SCRAPE_RETRY_COUNT: int = 3
    SCRAPE_RETRY_DELAY: int = 5
    SCRAPE_TIMEOUT: int = 30

    # =====================
    # CACHE SETTINGS
    # =====================
    CACHE_DIR: str = "./data/cache"
    CACHE_TTL_SECONDS: int = 3600
    CACHE_ENABLED: bool = True

    # =====================
    # CORS
    # =====================
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:8000", "*"]
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1", "*"]

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields in .env not defined here


# Initialize settings
settings = Settings()