import json
import shutil
from pathlib import Path

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - compatibility fallback
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_PROVIDER,
    RAW_DOCS_PATH,
    VECTORSTORE_DIR,
    VECTORSTORE_MANIFEST_PATH,
)
from embeddings import get_embeddings


def _load_docs(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Raw document file not found at '{path}'. Run scraper.py first.")

    docs: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of '{path}': {exc}") from exc
    return docs


def _reset_vectorstore(path: Path) -> None:
    if path.exists():
        if path.is_file() or path.is_symlink():
            path.unlink()
        else:
            shutil.rmtree(path)


def ingest() -> None:
    docs = _load_docs(RAW_DOCS_PATH)
    if not docs:
        raise RuntimeError("No raw documents found. Run scraper.py first.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )

    documents: list[Document] = []
    for doc in docs:
        text = doc.get("text", "").strip()
        if not text:
            continue

        metadata = {
            "source": doc.get("canonical_url") or doc.get("url", ""),
            "url": doc.get("url", ""),
            "canonical_url": doc.get("canonical_url") or doc.get("url", ""),
            "path": doc.get("path", ""),
            "page_type": doc.get("page_type", "unknown"),
            "title": doc.get("title", ""),
            "description": doc.get("description", ""),
            "headings": " > ".join(doc.get("headings", []) or []),
            "heading_count": len(doc.get("headings", []) or []),
            "fetched_at": doc.get("fetched_at", ""),
        }
        base_document = Document(page_content=text, metadata=metadata)
        chunks = splitter.split_documents([base_document])
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_char_count"] = len(chunk.page_content)
            documents.append(chunk)

    if not documents:
        raise RuntimeError("No chunks were produced from the scraped documents.")

    embeddings = get_embeddings()
    _reset_vectorstore(VECTORSTORE_DIR)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
        collection_name="debales_ai",
    )
    VECTORSTORE_MANIFEST_PATH.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "collection_name": "debales_ai",
                "embedding_provider": EMBEDDING_PROVIDER,
                "embedding_dimension": EMBEDDING_DIMENSION,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "page_count": len(docs),
                "chunk_count": len(documents),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Vector store saved to {VECTORSTORE_DIR} ({len(documents)} chunks from {len(docs)} pages)")


if __name__ == "__main__":
    ingest()
