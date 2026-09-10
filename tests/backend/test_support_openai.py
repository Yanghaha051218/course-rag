import json
from types import SimpleNamespace

import pytest

from course_rag_api.errors import ConfigurationError, SupportVerificationError
from course_rag_api.models import RetrievedChunk
from course_rag_api.support import OpenAISupportVerifier, SupportStatus


def _chunk(chunk_id: str = "chunk-1", text: str = "The Blueleaf coefficient is 7.25.") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        course_id="course-1",
        document_id="document-1",
        filename="lesson.md",
        text=text,
        score=0.8,
        chunk_index=0,
        source_type="section",
        source_start=1,
        source_end=1,
    )


def test_openai_verifier_uses_structured_response_with_only_supplied_evidence() -> None:
    calls = []
    response = SimpleNamespace(
        output_text=json.dumps(
            {
                "status": "SUPPORTED",
                "reason": "explicit_support",
                "supporting_chunk_ids": ["chunk-1"],
            }
        )
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **arguments: calls.append(arguments) or response
        )
    )
    verifier = OpenAISupportVerifier(
        api_key="test-key", model_name="gpt-test", client=client
    )

    decision = verifier.verify(
        question="What is the Blueleaf coefficient?",
        course_id="course-1",
        evidence=(_chunk(),),
    )

    assert decision.status is SupportStatus.SUPPORTED
    request = calls[0]
    assert request["model"] == "gpt-test"
    assert request["store"] is False
    assert "tools" not in request
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True
    assert "What is the Blueleaf coefficient?" in request["input"][0]["content"][0]["text"]
    assert "The Blueleaf coefficient is 7.25." in request["input"][0]["content"][0]["text"]
    assert "unrelated course text" not in request["input"][0]["content"][0]["text"]


def test_openai_verifier_rejects_missing_key_malformed_output_and_sdk_error() -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        OpenAISupportVerifier(api_key=None, model_name="gpt-test")

    malformed = OpenAISupportVerifier(
        api_key="test-key",
        model_name="gpt-test",
        client=SimpleNamespace(
            responses=SimpleNamespace(
                create=lambda **_: SimpleNamespace(output_text="not json")
            )
        ),
    )
    failing = OpenAISupportVerifier(
        api_key="test-key",
        model_name="gpt-test",
        client=SimpleNamespace(
            responses=SimpleNamespace(
                create=lambda **_: (_ for _ in ()).throw(RuntimeError("offline"))
            )
        ),
    )

    for verifier in (malformed, failing):
        with pytest.raises(SupportVerificationError):
            verifier.verify(
                question="What is the coefficient?",
                course_id="course-1",
                evidence=(_chunk(),),
            )


def test_openai_verifier_rejects_invalid_structured_status() -> None:
    verifier = OpenAISupportVerifier(
        api_key="test-key",
        model_name="gpt-test",
        client=SimpleNamespace(
            responses=SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    output_text=json.dumps(
                        {
                            "status": "NOT_A_STATUS",
                            "reason": "bad",
                            "supporting_chunk_ids": [],
                        }
                    )
                )
            )
        ),
    )

    with pytest.raises(SupportVerificationError, match="invalid support decision"):
        verifier.verify(
            question="What is the coefficient?",
            course_id="course-1",
            evidence=(_chunk(),),
        )
