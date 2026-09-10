from course_rag_api.config import Settings
from course_rag_api.embeddings import OpenAIEmbeddingProvider, create_embedding_provider
from course_rag_api.support import OpenAISupportVerifier, create_support_verifier


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


def test_openai_verifier_selection_uses_configured_model() -> None:
    settings = Settings(
        _env_file=None,
        OPENAI_API_KEY="test-key",
        verifier_provider="openai",
        verifier_model="gpt-test",
    )

    verifier = create_support_verifier(settings)

    assert isinstance(verifier, OpenAISupportVerifier)
    assert verifier.model_name == "gpt-test"
