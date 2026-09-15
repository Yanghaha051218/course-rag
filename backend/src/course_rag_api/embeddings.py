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


class FastEmbedEmbeddingProvider:
    """Local semantic embeddings loaded only when this provider is selected."""

    provider_name = "fastembed"
    expected_model = "BAAI/bge-small-en-v1.5"
    expected_dimension = 384

    def __init__(
        self, *, model_name: str, cache_dir: str, model: object | None = None
    ) -> None:
        if model_name != self.expected_model:
            raise ConfigurationError(
                f"FastEmbed model must be {self.expected_model} for M4-A.3"
            )
        self.model_name = model_name
        if model is None:
            try:
                from fastembed import TextEmbedding

                model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
            except ImportError as error:
                raise ConfigurationError("fastembed must be installed to use this provider") from error
            except Exception as error:
                raise ConfigurationError("FastEmbed model initialization failed") from error
        dimension = getattr(model, "embedding_size", None)
        if callable(dimension):
            dimension = dimension()
        if dimension != self.expected_dimension:
            raise ConfigurationError(
                f"FastEmbed model must report {self.expected_dimension} dimensions"
            )
        self.dimension = dimension
        self._model = model

    def _vectors(self, vectors: object, expected_count: int) -> list[list[float]]:
        try:
            result = [[float(value) for value in vector] for vector in vectors]
        except (TypeError, ValueError) as error:
            raise EmbeddingError("FastEmbed returned invalid embeddings") from error
        if len(result) != expected_count:
            raise EmbeddingError("FastEmbed returned an unexpected number of embeddings")
        if any(len(vector) != self.dimension for vector in result):
            raise EmbeddingError("FastEmbed returned an unexpected embedding dimension")
        return result

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise EmbeddingError("embedding input must not be empty")
        try:
            return self._vectors(self._model.embed(list(texts)), len(texts))
        except EmbeddingError:
            raise
        except Exception as error:
            raise EmbeddingError("FastEmbed embedding request failed") from error

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise EmbeddingError("embedding input must not be empty")
        try:
            return self._vectors(self._model.query_embed(text), 1)[0]
        except EmbeddingError:
            raise
        except Exception as error:
            raise EmbeddingError("FastEmbed query embedding failed") from error


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
    if settings.embedding_provider == "fastembed":
        if settings.embedding_model is not None:
            raise ConfigurationError("use COURSE_RAG_FASTEMBED_MODEL for FastEmbed")
        if settings.embedding_dimension not in (None, FastEmbedEmbeddingProvider.expected_dimension):
            raise ConfigurationError("FastEmbed embedding dimension must be 384")
        return FastEmbedEmbeddingProvider(
            model_name=settings.fastembed_model,
            cache_dir=str(settings.fastembed_cache_dir),
        )
    return OpenAIEmbeddingProvider(
        api_key=(
            settings.openai_api_key.get_secret_value()
            if settings.openai_api_key is not None
            else None
        ),
        model_name=settings.embedding_model or "text-embedding-3-small",
        dimension=settings.embedding_dimension or 1536,
    )
