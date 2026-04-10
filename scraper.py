import json
import logging
import re
from collections import deque
from datetime import datetime, timezone
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    ALLOWED_PATH_PREFIXES,
    BASE_URL,
    EXCLUDED_PATH_PREFIXES,
    MAX_PAGES,
    RAW_DOCS_PATH,
    REQUEST_TIMEOUT,
    SITEMAP_URL,
)


logger = logging.getLogger(__name__)

HEADING_TAGS = {"h1", "h2", "h3"}
STRUCTURAL_TAGS = {"p", "li", "blockquote"}
USER_AGENT = "DebalesAI-RAG-Bot/1.0 (+https://debales.ai)"


def _build_session() -> requests.Session:
    session = requests.Session()
    retry_kwargs = {
        "total": 3,
        "connect": 3,
        "read": 3,
        "backoff_factor": 0.6,
        "status_forcelist": (429, 500, 502, 503, 504),
        "raise_on_status": False,
    }
    try:
        retry = Retry(**retry_kwargs, allowed_methods=frozenset({"GET"}))
    except TypeError:  # pragma: no cover - older urllib3 compatibility
        retry = Retry(**retry_kwargs, method_whitelist=frozenset({"GET"}))
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _normalize_url(url: str) -> str | None:
    if not url:
        return None
    parsed = urlsplit(urljoin(BASE_URL, url))
    if parsed.scheme not in {"http", "https"}:
        return None

    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    canonical = urlunsplit((parsed.scheme, parsed.netloc.lower(), path or "/", "", ""))
    return canonical


def _matches_prefix(path: str, prefix: str) -> bool:
    if prefix == "/":
        return path == "/"
    normalized_prefix = prefix.rstrip("/")
    return path == normalized_prefix or path.startswith(f"{normalized_prefix}/")


def _is_excluded(path: str) -> bool:
    return any(_matches_prefix(path, prefix) for prefix in EXCLUDED_PATH_PREFIXES)


def _is_allowed_url(url: str) -> bool:
    normalized = _normalize_url(url)
    if not normalized:
        return False

    parsed = urlsplit(normalized)
    base = urlsplit(BASE_URL)
    if parsed.netloc != base.netloc:
        return False

    path = parsed.path or "/"
    if _is_excluded(path):
        return False

    if path == "/":
        return True

    return any(_matches_prefix(path, prefix) for prefix in ALLOWED_PATH_PREFIXES)


def _page_type_for_path(path: str) -> str:
    for prefix in ALLOWED_PATH_PREFIXES:
        if prefix == "/":
            continue
        if _matches_prefix(path, prefix):
            label = prefix.strip("/").split("/", 1)[0]
            if label in {"product", "products"}:
                return "product"
            if label in {"integration", "integrations"}:
                return "integration"
            if label in {"solution", "solutions", "use-cases", "case-studies"}:
                return "content"
            if label == "blog":
                return "blog"
            return label or "content"
    if path == "/":
        return "homepage"
    return "other"


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_page_data(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = _clean_text(soup.title.string)

    description_tag = soup.find("meta", attrs={"name": "description"})
    description = _clean_text(description_tag.get("content") or "") if description_tag else ""

    canonical_url = None
    for link_tag in soup.find_all("link", rel=True):
        rel_values = link_tag.get("rel", [])
        if isinstance(rel_values, str):
            rel_values = [rel_values]
        if "canonical" in rel_values:
            canonical_url = _normalize_url(link_tag.get("href") or "")
            break
    normalized_url = _normalize_url(url) or url
    if canonical_url and not _is_allowed_url(canonical_url):
        canonical_url = None

    headings: list[str] = []
    lines: list[str] = []
    if title:
        lines.append(f"# {title}")
    if description:
        lines.append(f"Description: {description}")

    seen_lines: set[str] = {value for value in {title, description} if value}
    for node in soup.find_all(list(HEADING_TAGS | STRUCTURAL_TAGS)):
        text = _clean_text(node.get_text(" ", strip=True))
        if not text or text in seen_lines:
            continue
        seen_lines.add(text)
        if node.name in HEADING_TAGS:
            level = int(node.name[1])
            headings.append(text)
            lines.append(f"{'#' * level} {text}")
        elif node.name == "li":
            lines.append(f"- {text}")
        elif node.name == "blockquote":
            lines.append(f"> {text}")
        else:
            lines.append(text)

    body_text = "\n\n".join(line.strip() for line in lines if line.strip())
    path = urlsplit(normalized_url).path or "/"
    page_type = _page_type_for_path(path)

    return {
        "url": normalized_url,
        "canonical_url": canonical_url or normalized_url,
        "path": path,
        "page_type": page_type,
        "title": title,
        "description": description,
        "headings": headings,
        "text": body_text,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def _fetch(session: requests.Session, url: str) -> str | None:
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        logger.warning("Skipping %s: %s", url, exc)
        return None


def _parse_sitemap(session: requests.Session, url: str, visited: set[str] | None = None) -> list[str]:
    visited = visited or set()
    normalized = _normalize_url(url)
    if not normalized or normalized in visited:
        return []
    visited.add(normalized)

    sitemap_xml = _fetch(session, normalized)
    if not sitemap_xml:
        return []

    soup = BeautifulSoup(sitemap_xml, "xml")
    urls: list[str] = []
    for loc in soup.find_all("loc"):
        value = _normalize_url(loc.text.strip())
        if not value:
            continue
        urls.append(value)
        if value.endswith(".xml"):
            urls.extend(_parse_sitemap(session, value, visited))

    return urls


def _seed_urls() -> list[str]:
    seeds = {BASE_URL}
    for prefix in ALLOWED_PATH_PREFIXES:
        if prefix == "/":
            continue
        seeds.add(urljoin(f"{BASE_URL}/", prefix.lstrip("/")))
    return sorted(seeds)


def crawl() -> list[dict]:
    session = _build_session()
    queue = deque()
    queued: set[str] = set()
    seen: set[str] = set()
    documents: list[dict] = []

    sitemap_urls = [url for url in _parse_sitemap(session, SITEMAP_URL) if _is_allowed_url(url)]
    seeds = sitemap_urls or _seed_urls()
    for seed in seeds:
        normalized = _normalize_url(seed)
        if normalized and normalized not in queued and _is_allowed_url(normalized):
            queue.append(normalized)
            queued.add(normalized)

    logger.info("Seeded crawl with %d URLs", len(queue))
    while queue and len(documents) < MAX_PAGES:
        url = queue.popleft()
        queued.discard(url)
        if url in seen:
            continue
        seen.add(url)

        html = _fetch(session, url)
        if not html:
            continue

        page = _extract_page_data(html, url)
        if page["text"]:
            documents.append(page)
        else:
            logger.info("Skipping empty page content: %s", url)

        soup = BeautifulSoup(html, "html.parser")
        for link in soup.find_all("a", href=True):
            next_url = _normalize_url(urljoin(url, link["href"]))
            if not next_url or next_url in seen or next_url in queued:
                continue
            if _is_allowed_url(next_url):
                queue.append(next_url)
                queued.add(next_url)

    logger.info(
        "Crawl finished: %d pages collected, %d URLs seen, %d queued",
        len(documents),
        len(seen),
        len(queue),
    )
    return documents


def save_documents(documents: list[dict]) -> None:
    RAW_DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAW_DOCS_PATH.open("w", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    docs = crawl()
    print(f"Collected {len(docs)} pages from {BASE_URL}")
    save_documents(docs)
    print(f"Saved to {RAW_DOCS_PATH}")
