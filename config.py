import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent


def _csv_env(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_path(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default))
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


BASE_URL = os.getenv("DEBALES_BASE_URL", "https://debales.ai").rstrip("/")
SITEMAP_URL = os.getenv("DEBALES_SITEMAP_URL", f"{BASE_URL}/sitemap.xml")
BASE_HOSTNAME = urlparse(BASE_URL).netloc

ALLOWED_PATH_PREFIXES = _csv_env(
    "ALLOWED_PATH_PREFIXES",
    [
        "/blog",
        "/docs",
        "/faq",
        "/logistics",
        "/ai-agent",
        "/ai-agents",
        "/case-study",
        "/case-studies",
        "/integration",
        "/integrations",
        "/product",
        "/products",
        "/solution",
        "/solutions",
        "/use-cases",
    ],
)
EXCLUDED_PATH_PREFIXES = _csv_env(
    "EXCLUDED_PATH_PREFIXES",
    [
        "/sign-in",
        "/sign-up",
        "/login",
        "/register",
        "/privacy",
        "/terms",
        "/contact",
    ],
)

MAX_PAGES = int(os.getenv("DEBALES_MAX_PAGES", "80"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "20"))
SERP_TIMEOUT = int(os.getenv("SERP_TIMEOUT", "20"))
SERP_MAX_RESULTS = int(os.getenv("SERP_MAX_RESULTS", "5"))

RAW_DOCS_PATH = os.getenv("RAW_DOCS_PATH", "data/raw/docs.jsonl")
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "data/vectorstore")
RAW_DOCS_PATH = _resolve_path("RAW_DOCS_PATH", RAW_DOCS_PATH)
VECTORSTORE_DIR = _resolve_path("VECTORSTORE_DIR", VECTORSTORE_DIR)
VECTORSTORE_MANIFEST_PATH = VECTORSTORE_DIR / "manifest.json"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_RELEVANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.18"))
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "auto").lower()
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("OPENROUTER_MODEL", os.getenv("LLM_MODEL", "openai/gpt-4o-mini"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
