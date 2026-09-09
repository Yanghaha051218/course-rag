import math
import re
from collections import Counter
from collections.abc import Sequence
from hashlib import blake2b
from typing import Protocol

from openai import OpenAI

from course_rag_api.config import Settings
from course_rag_api.errors import ConfigurationError, EmbeddingError


class EmbeddingProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_STOP_WORDS = frozenset(
    {"a", "an", "are", "is", "of", "the", "to", "what", "when", "where"}
)


class DeterministicEmbeddingProvider:
    """Small lexical hashing provider for offline tests, not semantic quality."""

    provider_name = "deterministic"
    model_name = "deterministic-hash-v1"

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        raw_tokens = [token.casefold() for token in _WORDS.findall(text)]
        if not raw_tokens:
            raise EmbeddingError("text must contain at least one word")
        tokens = [token for token in raw_tokens if token not in _STOP_WORDS]
        counts = Counter(tokens or raw_tokens)
        vector = [0.0] * self.dimension
        for token, count in counts.items():
            bucket = int.from_bytes(blake2b(token.encode(), digest_size=8).digest())
            vector[bucket % self.dimension] += float(count)
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OpenAIEmbeddingProvider:
    provider_name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None,
        model_name: str,
        dimension: int,
        client: OpenAI | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ConfigurationError(
                "OPENAI_API_KEY is required when the OpenAI embedding provider is selected"
            )
        if not model_name.strip():
            raise ConfigurationError("embedding model name must not be empty")
        if dimension <= 0:
            raise ConfigurationError("embedding dimension must be positive")
        self.model_name = model_name
        self.dimension = dimension
        self._client = client or OpenAI(api_key=api_key)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise EmbeddingError("embedding input must not be empty")
        try:
            response = self._client.embeddings.create(
                input=list(texts),
                model=self.model_name,
                dimensions=self.dimension,
                encoding_format="float",
            )
        except Exception as error:
            raise EmbeddingError("OpenAI embedding request failed") from error
        try:
            items = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in items] != list(range(len(texts))):
                raise ValueError("unexpected embedding indexes")
            vectors = [item.embedding for item in items]
        except (AttributeError, TypeError, ValueError) as error:
            raise EmbeddingError("OpenAI returned an invalid embedding response") from error
        if len(vectors) != len(texts):
            raise EmbeddingError("OpenAI returned an unexpected number of embeddings")
        if any(len(vector) != self.dimension for vector in vectors):
            raise EmbeddingError("OpenAI returned an unexpected embedding dimension")
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "deterministic":
        if settings.embedding_model not in (None, "deterministic-hash-v1"):
            raise ConfigurationError(
                "deterministic provider model must be deterministic-hash-v1"
            )
        return DeterministicEmbeddingProvider(settings.embedding_dimension or 256)
    return OpenAIEmbeddingProvider(
        api_key=(
            settings.openai_api_key.get_secret_value()
            if settings.openai_api_key is not None
            else None
        ),
        model_name=settings.embedding_model or "text-embedding-3-small",
        dimension=settings.embedding_dimension or 1536,
    )
