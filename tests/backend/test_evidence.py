from dataclasses import replace

import pytest

from course_rag_api.calibration import (
    CalibrationIdentity,
    LabeledScore,
    calibrate_threshold,
)
from course_rag_api.embeddings import DeterministicEmbeddingProvider
from course_rag_api.errors import CalibrationMismatchError
from course_rag_api.evidence import EvidenceGate
from course_rag_api.models import RetrievedChunk


def _calibration():
    identity = CalibrationIdentity(
        provider_name="deterministic",
        model_name="deterministic-hash-v1",
        dimension=256,
        dataset_version="m2b-v1",
        retrieval_limit=5,
        chunk_target_size=32,
        chunk_overlap=4,
    )
    return calibrate_threshold(
        (
            LabeledScore("answer", True, 0.8, 0.3),
            LabeledScore("unsupported", False, 0.4, 0.1),
        ),
        identity,
    )


def _chunk(score: float, course_id: str = "course-a") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-a",
        course_id=course_id,
        document_id="document-a",
        filename="lesson.md",
        text="The fictional valve opens after twelve minutes.",
        score=score,
        chunk_index=2,
        source_type="section",
        source_start=3,
        source_end=4,
    )


def test_evidence_gate_allows_strong_evidence_and_preserves_provenance() -> None:
    chunk = _chunk(0.9)

    decision = EvidenceGate(DeterministicEmbeddingProvider(256)).evaluate(
        course_id="course-a", retrieved_chunks=(chunk,), calibration=_calibration()
    )

    assert decision.status == "allow"
    assert decision.reason == "top1_score_meets_calibrated_threshold"
    assert decision.top_score == 0.9
    assert decision.evidence == (chunk,)
    assert not hasattr(decision, "answer")


def test_evidence_gate_abstains_for_weak_or_empty_evidence() -> None:
    gate = EvidenceGate(DeterministicEmbeddingProvider(256))
    calibration = _calibration()

    weak = gate.evaluate(
        course_id="course-a",
        retrieved_chunks=(_chunk(0.2),),
        calibration=calibration,
    )
    empty = gate.evaluate(
        course_id="course-a", retrieved_chunks=(), calibration=calibration
    )

    assert weak.status == "abstain"
    assert weak.reason == "insufficient_retrieval_evidence"
    assert weak.evidence == ()
    assert empty.status == "abstain"
    assert empty.reason == "no_evidence_retrieved"
    assert empty.top_score is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider_name", "openai"),
        ("model_name", "other-model"),
        ("dimension", 64),
    ],
)
def test_evidence_gate_rejects_mismatched_calibration_identity(
    field: str, value: str | int
) -> None:
    calibration = _calibration()
    bad_identity = replace(calibration.identity, **{field: value})

    with pytest.raises(CalibrationMismatchError, match=field):
        EvidenceGate(DeterministicEmbeddingProvider(256)).evaluate(
            course_id="course-a",
            retrieved_chunks=(_chunk(0.9),),
            calibration=replace(calibration, identity=bad_identity),
        )


def test_evidence_gate_never_returns_foreign_course_evidence() -> None:
    decision = EvidenceGate(DeterministicEmbeddingProvider(256)).evaluate(
        course_id="course-a",
        retrieved_chunks=(_chunk(0.9, course_id="course-b"),),
        calibration=_calibration(),
    )

    assert decision.status == "abstain"
    assert decision.reason == "foreign_course_evidence"
    assert decision.evidence == ()
