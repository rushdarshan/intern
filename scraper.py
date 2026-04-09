import json
import re
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from config import (
    ALLOWED_PATH_PREFIXES,
    BASE_URL,
    MAX_PAGES,
    RAW_DOCS_PATH,
    REQUEST_TIMEOUT,
    SITEMAP_URL,
)


def _is_allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    base = urlparse(BASE_URL)
    if parsed.netloc != base.netloc:
        return False
    path = parsed.path or "/"
    return any(path.startswith(prefix) for prefix in ALLOWED_PATH_PREFIXES)


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    text = " ".join(t.get_text(" ", strip=True) for t in soup.find_all(["p", "li", "h1", "h2", "h3"]))
    return title, _clean_text(text)


def _fetch(url: str) -> str | None:
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "DebalesAI-RAG-Bot/1.0"})
        resp.raise_for_status()
        return resp.text
    except requests.RequestException:
        return None


def _parse_sitemap(url: str) -> list[str]:
    sitemap_html = _fetch(url)
    if not sitemap_html:
        return []
    soup = BeautifulSoup(sitemap_html, "xml")
    return [loc.text.strip() for loc in soup.find_all("loc") if loc.text]


def crawl() -> list[dict]:
    urls = deque()
    seen = set()

    sitemap_urls = [u for u in _parse_sitemap(SITEMAP_URL) if _is_allowed_url(u)]
    if sitemap_urls:
        for url in sitemap_urls:
            urls.append(url)
    else:
        urls.append(BASE_URL)

    documents = []
    while urls and len(documents) < MAX_PAGES:
        url = urls.popleft()
        if url in seen:
            continue
        seen.add(url)

        html = _fetch(url)
        if not html:
            continue

        title, text = _extract_text(html)
        if text:
            documents.append({"url": url, "title": title, "text": text})

        soup = BeautifulSoup(html, "html.parser")
        for link in soup.find_all("a", href=True):
            next_url = urljoin(url, link["href"])
            if _is_allowed_url(next_url) and next_url not in seen:
                urls.append(next_url)

    return documents


def save_documents(documents: list[dict]) -> None:
    with open(RAW_DOCS_PATH, "w", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(json.dumps(doc, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    docs = crawl()
    print(f"Collected {len(docs)} pages from {BASE_URL}")
    save_documents(docs)
    print(f"Saved to {RAW_DOCS_PATH}")
