from dataclasses import replace
from datetime import datetime, timezone

import pytest

from course_rag_api.generation import (
    AnswerGenerationService,
    FinalStatus,
    GeneratedAnswer,
    GeneratedClaim,
    validate_citations,
)
from course_rag_api.models import Chunk, Document, RetrievedChunk
from course_rag_api.storage import SQLiteStore
from course_rag_api.support import (
    SupportDecision,
    SupportStatus,
    SupportVerificationService,
)


class StaticRetriever:
    def __init__(self, chunks: tuple[RetrievedChunk, ...]) -> None:
        self.chunks = chunks

    def retrieve(self, **_: object) -> tuple[RetrievedChunk, ...]:
        return self.chunks


class ScriptedVerifier:
    provider_name = "scripted"
    model_name = "support-test-v1"

    def __init__(self, decision: SupportDecision) -> None:
        self.decision = decision
        self.calls = 0

    def verify(self, **_: object) -> SupportDecision:
        self.calls += 1
        return self.decision


class ScriptedGenerator:
    provider_name = "scripted"
    model_name = "generation-test-v1"

    def __init__(self, answer: GeneratedAnswer | Exception) -> None:
        self.answer = answer
        self.calls = 0
        self.evidence: tuple[RetrievedChunk, ...] = ()

    def generate(
        self, *, question: str, verified_evidence: tuple[RetrievedChunk, ...]
    ) -> GeneratedAnswer:
        self.calls += 1
        self.evidence = verified_evidence
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _stored_chunks(
    tmp_path,
    texts: tuple[str, ...] = ("The Blueleaf coefficient is 7.25.",),
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


def _answer(*chunk_ids: str) -> GeneratedAnswer:
    claim = GeneratedClaim("The Blueleaf coefficient is 7.25.", chunk_ids)
    return GeneratedAnswer(claim.text, (claim,), "scripted", "generation-test-v1")


def test_generated_answer_requires_structured_supported_claims() -> None:
    answer = _answer("chunk-0")

    assert answer.answer == "The Blueleaf coefficient is 7.25."
    assert answer.claims[0].supporting_chunk_ids == ("chunk-0",)

    with pytest.raises(ValueError, match="claims"):
        GeneratedAnswer("Ungrounded prose.", (), "scripted", "generation-test-v1")
    with pytest.raises(ValueError, match="must consist"):
        GeneratedAnswer(
            "Extra unsupported prose.",
            (GeneratedClaim("A supported claim.", ("chunk-0",)),),
            "scripted",
            "generation-test-v1",
        )


def test_generation_runs_only_for_supported_evidence_and_returns_citations(
    tmp_path,
) -> None:
    store, course_id, chunks = _stored_chunks(
        tmp_path,
        ("The Blueleaf coefficient is 7.25.", "Unverified nearby context."),
    )
    verifier = ScriptedVerifier(
        SupportDecision(
            SupportStatus.SUPPORTED,
            "explicit_support",
            ("chunk-0",),
            verifier_provider="scripted",
            verifier_model="support-test-v1",
        )
    )
    generator = ScriptedGenerator(_answer("chunk-0"))
    support = SupportVerificationService(store, StaticRetriever(chunks), verifier)

    response = AnswerGenerationService(support, generator).answer_question(
        course_id=course_id, question="What is the Blueleaf coefficient?"
    )

    assert response.status is FinalStatus.ANSWERED
    assert response.answer == "The Blueleaf coefficient is 7.25."
    assert response.citations[0].chunk_id == "chunk-0"
    assert response.citations[0].filename == "lesson.md"
    assert generator.calls == 1
    assert generator.evidence == (chunks[0],)


@pytest.mark.parametrize(
    ("status", "reason", "has_evidence"),
    [
        (SupportStatus.INSUFFICIENT, "missing_attribute", True),
        (SupportStatus.CONFLICTING, "incompatible_facts", True),
        (SupportStatus.SUPPORTED, "explicit_support", False),
    ],
)
def test_non_supported_or_empty_evidence_never_calls_generator(
    tmp_path, status: SupportStatus, reason: str, has_evidence: bool
) -> None:
    store, course_id, chunks = _stored_chunks(tmp_path)
    verifier = ScriptedVerifier(
        SupportDecision(
            status,
            reason,
            ("chunk-0",) if status is SupportStatus.SUPPORTED else (),
            "scripted",
            "support-test-v1",
        )
    )
    generator = ScriptedGenerator(_answer("chunk-0"))
    support = SupportVerificationService(
        store, StaticRetriever(chunks if has_evidence else ()), verifier
    )

    response = AnswerGenerationService(support, generator).answer_question(
        course_id=course_id, question="What is the coefficient?"
    )

    assert response.status is FinalStatus.ABSTAINED
    assert response.answer is None
    assert generator.calls == 0


def test_citation_validation_rejects_invented_foreign_or_missing_bindings(tmp_path) -> None:
    _, course_id, chunks = _stored_chunks(tmp_path)
    foreign = replace(chunks[0], course_id="foreign-course")

    assert validate_citations(_answer("chunk-0"), chunks, course_id)[0].chunk_id == "chunk-0"
    with pytest.raises(ValueError, match="unknown"):
        validate_citations(_answer("invented"), chunks, course_id)
    with pytest.raises(ValueError, match="foreign"):
        validate_citations(_answer("chunk-0"), (foreign,), course_id)
    with pytest.raises(ValueError, match="supporting"):
        GeneratedClaim("Ungrounded claim.", ())


def test_generation_exception_or_invalid_citation_fails_closed(tmp_path) -> None:
    store, course_id, chunks = _stored_chunks(tmp_path)
    verifier = ScriptedVerifier(
        SupportDecision(
            SupportStatus.SUPPORTED,
            "explicit_support",
            ("chunk-0",),
            "scripted",
            "support-test-v1",
        )
    )
    support = SupportVerificationService(store, StaticRetriever(chunks), verifier)
    error = AnswerGenerationService(
        support, ScriptedGenerator(RuntimeError("offline"))
    ).answer_question(course_id=course_id, question="What is the coefficient?")
    invalid = AnswerGenerationService(
        support, ScriptedGenerator(_answer("invented"))
    ).answer_question(course_id=course_id, question="What is the coefficient?")

    assert (error.status, error.reason) == (FinalStatus.ABSTAINED, "generator_error")
    assert (invalid.status, invalid.reason) == (
        FinalStatus.ABSTAINED,
        "citation_validation_failed",
    )
