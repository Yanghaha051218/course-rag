from datetime import datetime, timezone
from dataclasses import replace

import pytest

from course_rag_api.models import Chunk, Document, RetrievedChunk
from course_rag_api.storage import SQLiteStore
from course_rag_api.support import (
    SupportDecision,
    SupportStatus,
    SupportVerificationService,
)


class ScriptedSupportVerifier:
    provider_name = "scripted"
    model_name = "test-v1"

    def __init__(self, result: SupportDecision | Exception) -> None:
        self.result = result
        self.calls = 0

    def verify(self, **_: object) -> SupportDecision:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class StaticRetriever:
    def __init__(self, chunks: tuple[RetrievedChunk, ...]) -> None:
        self.chunks = chunks

    def retrieve(self, **_: object) -> tuple[RetrievedChunk, ...]:
        return self.chunks


def _stored_chunks(
    tmp_path, texts: tuple[str, ...] = ("The Blueleaf coefficient is 7.25.",)
) -> tuple[SQLiteStore, str, tuple[RetrievedChunk, ...]]:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    course = store.create_course("Synthetic Course")
    created_at = datetime.now(timezone.utc).isoformat()
    document = Document(
        id="document-a",
        course_id=course.id,
        filename="lesson.md",
        file_type="markdown",
        source_path="lesson.md",
        checksum="checksum-a",
        page_or_unit_count=len(texts),
        created_at=created_at,
    )
    chunks = tuple(
        Chunk(
            id=f"chunk-{index}",
            course_id=course.id,
            document_id=document.id,
            text=text,
            chunk_index=index,
            source_type="section",
            source_start=index + 1,
            source_end=index + 1,
            created_at=created_at,
        )
        for index, text in enumerate(texts)
    )
    store.persist_document(document, chunks)
    retrieved = tuple(
        RetrievedChunk(
            chunk_id=chunk.id,
            course_id=chunk.course_id,
            document_id=chunk.document_id,
            filename=document.filename,
            text=chunk.text,
            score=0.9 - index / 10,
            chunk_index=chunk.chunk_index,
            source_type=chunk.source_type,
            source_start=chunk.source_start,
            source_end=chunk.source_end,
        )
        for index, chunk in enumerate(chunks)
    )
    return store, course.id, retrieved


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        (
            {
                "status": "SUPPORTED",
                "reason": "explicit_support",
                "supporting_chunk_ids": ["chunk-0"],
            },
            SupportStatus.SUPPORTED,
        ),
        (
            {
                "status": "INSUFFICIENT",
                "reason": "missing_attribute",
                "supporting_chunk_ids": [],
            },
            SupportStatus.INSUFFICIENT,
        ),
        (
            {
                "status": "CONFLICTING",
                "reason": "incompatible_facts",
                "supporting_chunk_ids": ["chunk-0", "chunk-1"],
            },
            SupportStatus.CONFLICTING,
        ),
    ],
)
def test_support_decision_accepts_the_three_explicit_statuses(payload, status) -> None:
    decision = SupportDecision.from_payload(
        payload, verifier_provider="scripted", verifier_model="test-v1"
    )

    assert decision.status is status


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "UNKNOWN", "reason": "bad", "supporting_chunk_ids": []},
        {"status": "SUPPORTED", "reason": "bad", "supporting_chunk_ids": []},
        {
            "status": "INSUFFICIENT",
            "reason": "bad",
            "supporting_chunk_ids": ["chunk-0"],
        },
        {
            "status": "SUPPORTED",
            "reason": "bad",
            "supporting_chunk_ids": ["chunk-0", "chunk-0"],
        },
    ],
)
def test_support_decision_rejects_malformed_states(payload) -> None:
    with pytest.raises(ValueError):
        SupportDecision.from_payload(
            payload, verifier_provider="scripted", verifier_model="test-v1"
        )

    with pytest.raises(ValueError, match="invalid support status"):
        SupportDecision(
            "SUPPORTED",  # type: ignore[arg-type]
            "explicit_support",
            ("chunk-0",),
            "scripted",
            "test-v1",
        )


def test_service_returns_insufficient_without_calling_verifier_for_no_evidence(
    tmp_path,
) -> None:
    store, course_id, _ = _stored_chunks(tmp_path)
    verifier = ScriptedSupportVerifier(
        SupportDecision(
            SupportStatus.SUPPORTED,
            "explicit_support",
            ("chunk-0",),
            verifier.provider_name if False else "scripted",
            "test-v1",
        )
    )

    decision = SupportVerificationService(
        store, StaticRetriever(()), verifier
    ).verify_question_support(course_id=course_id, question="What is the coefficient?")

    assert (decision.status, decision.reason, verifier.calls) == (
        SupportStatus.INSUFFICIENT,
        "no_evidence_retrieved",
        0,
    )


def test_service_preserves_valid_multi_chunk_support(tmp_path) -> None:
    store, course_id, chunks = _stored_chunks(
        tmp_path,
        (
            "Solar vines receive three hours of ultraviolet exposure.",
            "The exposure occurs before artificial dusk.",
        ),
    )
    verifier = ScriptedSupportVerifier(
        SupportDecision(
            SupportStatus.SUPPORTED,
            "explicit_multi_chunk_support",
            ("chunk-0", "chunk-1"),
            "scripted",
            "test-v1",
        )
    )

    decision = SupportVerificationService(
        store, StaticRetriever(chunks), verifier
    ).verify_question_support(
        course_id=course_id,
        question="How long are solar vines exposed and when does it occur?",
    )

    assert decision.status is SupportStatus.SUPPORTED
    assert decision.supporting_chunk_ids == ("chunk-0", "chunk-1")
    assert verifier.calls == 1


def test_service_rejects_foreign_evidence_without_calling_verifier(tmp_path) -> None:
    store, course_id, chunks = _stored_chunks(tmp_path)
    foreign = replace(chunks[0], course_id="foreign")
    verifier = ScriptedSupportVerifier(
        SupportDecision(
            SupportStatus.SUPPORTED,
            "explicit_support",
            ("chunk-0",),
            "scripted",
            "test-v1",
        )
    )

    decision = SupportVerificationService(
        store, StaticRetriever((foreign,)), verifier
    ).verify_question_support(course_id=course_id, question="What is the coefficient?")

    assert (decision.status, decision.reason, verifier.calls) == (
        SupportStatus.INSUFFICIENT,
        "foreign_course_evidence",
        0,
    )


def test_service_fails_closed_for_verifier_exception_or_invented_chunk(tmp_path) -> None:
    store, course_id, chunks = _stored_chunks(tmp_path)
    exception_verifier = ScriptedSupportVerifier(RuntimeError("offline"))
    invented_verifier = ScriptedSupportVerifier(
        SupportDecision(
            SupportStatus.SUPPORTED,
            "explicit_support",
            ("invented-chunk",),
            "scripted",
            "test-v1",
        )
    )

    exception = SupportVerificationService(
        store, StaticRetriever(chunks), exception_verifier
    ).verify_question_support(course_id=course_id, question="What is the coefficient?")
    invented = SupportVerificationService(
        store, StaticRetriever(chunks), invented_verifier
    ).verify_question_support(course_id=course_id, question="What is the coefficient?")

    assert (exception.status, exception.reason) == (
        SupportStatus.INSUFFICIENT,
        "verifier_error",
    )
    assert (invented.status, invented.reason) == (
        SupportStatus.INSUFFICIENT,
        "invalid_supporting_chunk_ids",
    )


def test_service_preserves_same_course_conflict_without_generation(tmp_path) -> None:
    store, course_id, chunks = _stored_chunks(
        tmp_path,
        ("The Helios pressure is 81 kPa.", "The Helios pressure is 93 kPa."),
    )
    verifier = ScriptedSupportVerifier(
        SupportDecision(
            SupportStatus.CONFLICTING,
            "incompatible_candidate_answers",
            ("chunk-0", "chunk-1"),
            "scripted",
            "test-v1",
        )
    )

    decision = SupportVerificationService(
        store, StaticRetriever(chunks), verifier
    ).verify_question_support(course_id=course_id, question="What is the Helios pressure?")

    assert decision.status is SupportStatus.CONFLICTING
    assert decision.supporting_chunk_ids == ("chunk-0", "chunk-1")


@pytest.mark.parametrize(
    ("category", "question", "status"),
    [
        ("exact answer", "What is the Blueleaf coefficient?", SupportStatus.SUPPORTED),
        ("paraphrase", "How long is the ultraviolet exposure?", SupportStatus.SUPPORTED),
        ("near miss", "Who discovered the Blueleaf coefficient?", SupportStatus.INSUFFICIENT),
        ("wrong attribute", "When was the coefficient discovered?", SupportStatus.INSUFFICIENT),
        ("lexical overlap", "What ultraviolet wavelength is used?", SupportStatus.INSUFFICIENT),
    ],
)
def test_scripted_semantic_policy_fixtures_are_fail_closed(
    tmp_path, category: str, question: str, status: SupportStatus
) -> None:
    store, course_id, chunks = _stored_chunks(
        tmp_path, ("Solar vines receive three hours of ultraviolet exposure.",)
    )
    decision = SupportDecision(
        status,
        "scripted_policy_fixture",
        ("chunk-0",) if status is SupportStatus.SUPPORTED else (),
        "scripted",
        "test-v1",
    )
    verifier = ScriptedSupportVerifier(decision)

    result = SupportVerificationService(
        store, StaticRetriever(chunks), verifier
    ).verify_question_support(course_id=course_id, question=question)

    assert result.status is status, category
