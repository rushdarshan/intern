import os

from dotenv import load_dotenv


load_dotenv()


BASE_URL = os.getenv("DEBALES_BASE_URL", "https://debales.ai")
SITEMAP_URL = os.getenv("DEBALES_SITEMAP_URL", f"{BASE_URL.rstrip('/')}/sitemap.xml")

ALLOWED_PATH_PREFIXES = [
    "/",
    "/blog",
    "/product",
    "/products",
    "/integration",
    "/integrations",
    "/solutions",
    "/use-cases",
]

MAX_PAGES = int(os.getenv("DEBALES_MAX_PAGES", "80"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "20"))

RAW_DOCS_PATH = os.getenv("RAW_DOCS_PATH", "data/raw/docs.jsonl")
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "data/vectorstore")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("OPENROUTER_MODEL", os.getenv("LLM_MODEL", "openai/gpt-4o-mini"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
