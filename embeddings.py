import hashlib
import logging
import math
import os
import re
from functools import lru_cache

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from langchain_core.embeddings import Embeddings
except ImportError:  # pragma: no cover - compatibility fallback
    from langchain.schema.embeddings import Embeddings  # type: ignore

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover - optional dependency fallback
    from langchain_community.embeddings import HuggingFaceEmbeddings

from config import EMBEDDING_DEVICE, EMBEDDING_DIMENSION, EMBEDDING_MODEL, EMBEDDING_PROVIDER


logger = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")


class HashingEmbeddings(Embeddings):
    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest, "big") % self.dimension
        sign = -1.0 if digest[0] & 1 else 1.0
        return bucket, sign

    def _vectorize(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = TOKEN_PATTERN.findall(text.lower())
        if not tokens:
            return vector

        for token in tokens:
            bucket, sign = self._bucket(token)
            vector[bucket] += sign

        for left, right in zip(tokens, tokens[1:]):
            bucket, sign = self._bucket(f"{left}_{right}")
            vector[bucket] += sign * 0.5

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)


def _load_huggingface_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        show_progress=False,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_embeddings():
    provider = EMBEDDING_PROVIDER
    if provider == "hf":
        try:
            return _load_huggingface_embeddings()
        except Exception as exc:  # pragma: no cover - network/model availability dependent
            logger.warning("Falling back to local hashing embeddings because Hugging Face embeddings failed: %s", exc)
            return HashingEmbeddings(dimension=EMBEDDING_DIMENSION)

    if provider not in {"hash", "hashed", "local", "auto"}:
        logger.warning("Unknown EMBEDDING_PROVIDER '%s'; using local hashing embeddings.", provider)
        return HashingEmbeddings(dimension=EMBEDDING_DIMENSION)

    if provider == "auto":
        try:
            return _load_huggingface_embeddings()
        except Exception as exc:  # pragma: no cover - network/model availability dependent
            logger.warning("Auto embedding selection fell back to local hashing embeddings: %s", exc)

    return HashingEmbeddings(dimension=EMBEDDING_DIMENSION)
