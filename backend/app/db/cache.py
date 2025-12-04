# backend/app/db/cache.py

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class SimpleCache:
    """Basic caching"""

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or settings.CACHE_DIR
        self.ttl = settings.CACHE_TTL_SECONDS
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def _get_cache_file(self, key: str) -> str:
        """Cache from convo key"""
        safe_key = "".join(c if c.isalnum() else "_" for c in key)
        return os.path.join(self.cache_dir, f"{safe_key}.json")

    def get(self, key: str):
        """return value from cache"""
        try:
            cache_file = self._get_cache_file(key)
            if not os.path.exists(cache_file):
                return None

            with open(cache_file, 'r') as f:
                data = json.load(f)

            # Checkss if expired
            created_at = datetime.fromisoformat(data['created_at'])
            if datetime.utcnow() - created_at > timedelta(seconds=self.ttl):
                os.remove(cache_file)
                return None

            return data['value']
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None

    def set(self, key: str, value):
        """store value in cache"""
        try:
            cache_file = self._get_cache_file(key)
            data = {
                'value': value,
                'created_at': datetime.utcnow().isoformat()
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Cache error: {str(e)}")

    def delete(self, key: str):
        """delete from cache"""
        try:
            cache_file = self._get_cache_file(key)
            if os.path.exists(cache_file):
                os.remove(cache_file)
        except Exception as e:
            logger.error(f"Cache delete error: {str(e)}")

    def clear(self):
        """Clear all cache"""
        try:
            for file in os.listdir(self.cache_dir):
                if file.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, file))
        except Exception as e:
            logger.error(f"Cache clear error: {str(e)}")


# Global cache instance
cache = SimpleCache()
