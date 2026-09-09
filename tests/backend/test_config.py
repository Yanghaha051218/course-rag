from course_rag_api.config import Settings
from course_rag_api.embeddings import OpenAIEmbeddingProvider, create_embedding_provider


def test_settings_accept_standard_openai_api_key(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-standard-environment")

    settings = Settings(_env_file=None)

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "from-standard-environment"


def test_openai_provider_selection_uses_openai_defaults() -> None:
    settings = Settings(
        _env_file=None,
        OPENAI_API_KEY="test-key",
        embedding_provider="openai",
    )

    provider = create_embedding_provider(settings)

    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert provider.model_name == "text-embedding-3-small"
    assert provider.dimension == 1536
