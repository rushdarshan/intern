import json
from functools import lru_cache

from langchain_community.vectorstores import Chroma

from config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_PROVIDER,
    RAG_RELEVANCE_THRESHOLD,
    RAG_TOP_K,
    VECTORSTORE_DIR,
    VECTORSTORE_MANIFEST_PATH,
)
from embeddings import get_embeddings


def _load_manifest() -> dict:
    if not VECTORSTORE_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Vector store manifest not found at '{VECTORSTORE_MANIFEST_PATH}'. Run `python ingest.py` to rebuild the index."
        )

    with VECTORSTORE_MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    expected = {
        "schema_version": 1,
        "collection_name": "debales_ai",
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_dimension": EMBEDDING_DIMENSION,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"Vector store manifest mismatch for '{key}': expected {value!r}, found {manifest.get(key)!r}. "
                "Run `python ingest.py` to rebuild the index."
            )

    return manifest


@lru_cache(maxsize=1)
def get_store():
    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found at '{VECTORSTORE_DIR}'. Run `python scraper.py` and `python ingest.py` first."
        )

    sqlite_path = VECTORSTORE_DIR / "chroma.sqlite3"
    if not sqlite_path.exists():
        raise FileNotFoundError(
            f"Chroma database file not found at '{sqlite_path}'. Rebuild the vector store with `python ingest.py`."
        )

    _load_manifest()
    embeddings = get_embeddings()
    try:
        store = Chroma(
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=embeddings,
            collection_name="debales_ai",
        )
        chunk_count = store._collection.count()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load the vector store at '{VECTORSTORE_DIR}'. Rebuild it with `python ingest.py`."
        ) from exc

    if chunk_count == 0:
        raise RuntimeError(
            f"The vector store at '{VECTORSTORE_DIR}' is empty. Run `python scraper.py` and `python ingest.py` again."
        )

    return store


def get_retriever():
    if RAG_TOP_K < 1:
        raise ValueError(f"RAG_TOP_K must be at least 1, got {RAG_TOP_K}.")

    store = get_store()
    return store.as_retriever(search_kwargs={"k": RAG_TOP_K})


def search_relevant_chunks(query: str):
    if RAG_TOP_K < 1:
        raise ValueError(f"RAG_TOP_K must be at least 1, got {RAG_TOP_K}.")

    store = get_store()
    matches = store.similarity_search_with_relevance_scores(query, k=RAG_TOP_K)
    return [(doc, score) for doc, score in matches if score >= RAG_RELEVANCE_THRESHOLD]
