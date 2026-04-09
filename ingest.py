import json
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from config import CHUNK_OVERLAP, CHUNK_SIZE, RAW_DOCS_PATH, VECTORSTORE_DIR
from embeddings import get_embeddings


def _load_docs(path: str) -> list[dict]:
    docs = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def ingest() -> None:
    docs = _load_docs(RAW_DOCS_PATH)
    if not docs:
        raise RuntimeError("No raw documents found. Run scraper.py first.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = []
    metadatas = []
    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({"source": doc["url"], "title": doc.get("title", "")})

    embeddings = get_embeddings()
    Path(VECTORSTORE_DIR).mkdir(parents=True, exist_ok=True)
    Chroma.from_texts(texts, embeddings, metadatas=metadatas, persist_directory=VECTORSTORE_DIR)
    print(f"Vector store saved to {VECTORSTORE_DIR}")


if __name__ == "__main__":
    ingest()
