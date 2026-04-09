import logging
import os

from dotenv import load_dotenv
import requests

load_dotenv()

logger = logging.getLogger(__name__)


def serp_search(query: str, num_results: int = 5) -> list[dict]:
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        logger.error("SERPAPI_API_KEY is missing.")
        raise RuntimeError("SERPAPI_API_KEY is missing.")

    logger.info(f"🔍 Searching SerpAPI for: {query}")
    params = {
        "engine": "google",
        "q": query,
        "num": num_results,
        "api_key": api_key,
    }
    try:
        response = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
        response.raise_for_status()
        results = response.json()
        organic = results.get("organic_results", []) or []
        logger.info(f"✓ Found {len(organic)} organic results")

        cleaned = []
        for item in organic[:num_results]:
            cleaned.append(
                {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                }
            )
        logger.info(f"✓ Cleaned {len(cleaned)} results for return")
        return cleaned
    except Exception as e:
        logger.error(f"❌ SerpAPI error: {e}")
        raise
