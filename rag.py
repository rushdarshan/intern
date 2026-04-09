from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import Chroma

from config import RAG_TOP_K, VECTORSTORE_DIR
from embeddings import get_embeddings


@lru_cache(maxsize=1)
def _get_store():
    if not Path(VECTORSTORE_DIR).exists():
        raise FileNotFoundError(
            f"Vector store not found at '{VECTORSTORE_DIR}'. Run `python scraper.py` and `python ingest.py` first."
        )
    embeddings = get_embeddings()
    return Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)


def get_retriever():
    store = _get_store()
    return store.as_retriever(search_kwargs={"k": RAG_TOP_K})
