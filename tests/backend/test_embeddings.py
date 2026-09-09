from types import SimpleNamespace

import pytest

from course_rag_api.embeddings import (
    DeterministicEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from course_rag_api.errors import ConfigurationError, EmbeddingError


def _similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def test_deterministic_provider_is_stable_and_lexically_meaningful() -> None:
    provider = DeterministicEmbeddingProvider(dimension=64)
    blueleaf = provider.embed_query("Blueleaf coefficient")

    assert provider.dimension == 64
    assert provider.model_name == "deterministic-hash-v1"
    assert blueleaf == provider.embed_documents(["Blueleaf coefficient"])[0]
    assert _similarity(
        blueleaf, provider.embed_query("What is the Blueleaf coefficient?")
    ) > _similarity(blueleaf, provider.embed_query("station pressure"))


def test_openai_provider_requires_key_and_uses_batch_embeddings() -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingProvider(
            api_key=None, model_name="text-embedding-3-small", dimension=3
        )
    with pytest.raises(ConfigurationError, match="model name"):
        OpenAIEmbeddingProvider(api_key="test-key", model_name=" ", dimension=3)

    response = SimpleNamespace(
        data=[
            SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0]),
            SimpleNamespace(index=1, embedding=[0.0, 1.0, 0.0]),
        ]
    )
    calls = []

    def create(**arguments):
        calls.append(arguments)
        return response

    embeddings = SimpleNamespace(create=create)
    client = SimpleNamespace(embeddings=embeddings)
    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        model_name="text-embedding-3-small",
        dimension=3,
        client=client,
    )

    assert provider.embed_documents(["first", "second"]) == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    assert calls == [
        {
            "input": ["first", "second"],
            "model": "text-embedding-3-small",
            "dimensions": 3,
            "encoding_format": "float",
        }
    ]


def test_openai_provider_surfaces_invalid_response_dimension() -> None:
    response = SimpleNamespace(
        data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])]
    )
    client = SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda **_: response)
    )
    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        model_name="text-embedding-3-small",
        dimension=3,
        client=client,
    )

    with pytest.raises(EmbeddingError, match="dimension"):
        provider.embed_query("query")


def test_openai_provider_rejects_misaligned_response_indexes() -> None:
    response = SimpleNamespace(
        data=[SimpleNamespace(index=1, embedding=[1.0, 0.0, 0.0])]
    )
    client = SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda **_: response)
    )
    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        model_name="text-embedding-3-small",
        dimension=3,
        client=client,
    )

    with pytest.raises(EmbeddingError, match="invalid embedding response"):
        provider.embed_query("query")


def test_openai_provider_surfaces_sdk_failure_without_network() -> None:
    def fail(**_):
        raise RuntimeError("simulated SDK failure")

    client = SimpleNamespace(embeddings=SimpleNamespace(create=fail))
    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        model_name="text-embedding-3-small",
        dimension=3,
        client=client,
    )

    with pytest.raises(EmbeddingError, match="request failed"):
        provider.embed_query("query")
