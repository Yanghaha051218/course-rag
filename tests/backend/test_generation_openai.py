import json
from types import SimpleNamespace

import pytest

from course_rag_api.errors import ConfigurationError, GenerationError
from course_rag_api.generation import OpenAIGenerator
from course_rag_api.models import RetrievedChunk


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        course_id="course-1",
        document_id="document-1",
        filename="lesson.md",
        text="The Blueleaf coefficient is 7.25.",
        score=0.8,
        chunk_index=0,
        source_type="section",
        source_start=1,
        source_end=1,
    )


def test_openai_generator_uses_strict_structured_output_with_verified_evidence() -> None:
    calls = []
    response = SimpleNamespace(
        output_text=json.dumps(
            {
                "answer": "The Blueleaf coefficient is 7.25.",
                "claims": [
                    {
                        "text": "The Blueleaf coefficient is 7.25.",
                        "supporting_chunk_ids": ["chunk-1"],
                    }
                ],
            }
        )
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **arguments: calls.append(arguments) or response
        )
    )
    generator = OpenAIGenerator(api_key="test-key", model_name="gpt-test", client=client)

    answer = generator.generate(
        question="What is the Blueleaf coefficient?", verified_evidence=(_chunk(),)
    )

    assert answer.claims[0].supporting_chunk_ids == ("chunk-1",)
    request = calls[0]
    assert request["store"] is False
    assert "tools" not in request
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True
    request_text = request["input"][0]["content"][0]["text"]
    assert "What is the Blueleaf coefficient?" in request_text
    assert "The Blueleaf coefficient is 7.25." in request_text
    assert "unverified evidence" not in request_text


def test_openai_generator_rejects_missing_key_and_malformed_output() -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        OpenAIGenerator(api_key=None, model_name="gpt-test")

    generator = OpenAIGenerator(
        api_key="test-key",
        model_name="gpt-test",
        client=SimpleNamespace(
            responses=SimpleNamespace(
                create=lambda **_: SimpleNamespace(output_text="not json")
            )
        ),
    )

    with pytest.raises(GenerationError, match="invalid generated answer"):
        generator.generate(question="What is the coefficient?", verified_evidence=(_chunk(),))
    with pytest.raises(GenerationError, match="verified evidence"):
        generator.generate(question="What is the coefficient?", verified_evidence=())


def test_openai_generator_surfaces_sdk_error_without_network() -> None:
    generator = OpenAIGenerator(
        api_key="test-key",
        model_name="gpt-test",
        client=SimpleNamespace(
            responses=SimpleNamespace(
                create=lambda **_: (_ for _ in ()).throw(RuntimeError("offline"))
            )
        ),
    )

    with pytest.raises(GenerationError, match="generation request failed"):
        generator.generate(question="What is the coefficient?", verified_evidence=(_chunk(),))
