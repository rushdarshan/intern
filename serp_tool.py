import logging
import os
import re
from typing import TypedDict

from dotenv import load_dotenv
import requests

from config import SERP_MAX_RESULTS, SERP_TIMEOUT

load_dotenv()

logger = logging.getLogger(__name__)


def _redact_sensitive_text(text: str) -> str:
    return re.sub(r"api_key=([^&\s]+)", "api_key=[redacted]", text)


class SerpResult(TypedDict):
    title: str
    link: str
    snippet: str
    position: int


class SerpResponse(TypedDict):
    results: list[SerpResult]
    error: str | None


def serp_search(query: str, num_results: int = 5) -> SerpResponse:
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        message = "SERPAPI_API_KEY is missing."
        logger.error(message)
        return {"results": [], "error": message}

    logger.info("Searching SerpAPI for: %s", query)
    params = {
        "engine": "google",
        "q": query,
        "num": min(num_results, SERP_MAX_RESULTS),
        "api_key": api_key,
    }

    try:
        response = requests.get("https://serpapi.com/search.json", params=params, timeout=SERP_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        message = _redact_sensitive_text(f"SerpAPI request failed: {exc}")
        logger.warning(message)
        return {"results": [], "error": message}
    except ValueError as exc:
        message = _redact_sensitive_text(f"SerpAPI returned invalid JSON: {exc}")
        logger.warning(message)
        return {"results": [], "error": message}

    cleaned: list[SerpResult] = []
    answer_box = payload.get("answer_box") or {}
    if answer_box:
        snippet = answer_box.get("answer") or answer_box.get("snippet") or answer_box.get("title") or ""
        if snippet:
            cleaned.append(
                {
                    "title": (answer_box.get("title") or "Direct answer").strip(),
                    "link": (answer_box.get("link") or "").strip(),
                    "snippet": snippet.strip(),
                    "position": 0,
                }
            )

    knowledge_graph = payload.get("knowledge_graph") or {}
    description = knowledge_graph.get("description")
    if description:
        cleaned.append(
            {
                "title": (knowledge_graph.get("title") or "Knowledge graph").strip(),
                "link": (knowledge_graph.get("website") or "").strip(),
                "snippet": description.strip(),
                "position": 0,
            }
        )

    organic = payload.get("organic_results", []) or []
    for index, item in enumerate(organic[:num_results], start=1):
        snippet = item.get("snippet") or ""
        cleaned.append(
            {
                "title": item.get("title", "").strip(),
                "link": item.get("link", "").strip(),
                "snippet": snippet.strip(),
                "position": index,
            }
        )

    deduped: list[SerpResult] = []
    seen = set()
    for item in cleaned:
        key = (item["title"], item["link"], item["snippet"])
        if not any(key) or key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    logger.info("Retrieved %s cleaned SerpAPI results", len(deduped))
    return {"results": deduped[:num_results], "error": None}
