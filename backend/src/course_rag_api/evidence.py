from dataclasses import dataclass
from typing import Literal, Sequence

from course_rag_api.calibration import CalibrationResult
from course_rag_api.embeddings import EmbeddingProvider
from course_rag_api.errors import CalibrationMismatchError
from course_rag_api.models import RetrievedChunk


@dataclass(frozen=True, slots=True)
class EvidenceDecision:
    status: Literal["allow", "abstain"]
    reason: str
    top_score: float | None
    threshold: float
    evidence: tuple[RetrievedChunk, ...]


class EvidenceGate:
    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider

    def evaluate(
        self,
        *,
        course_id: str,
        retrieved_chunks: Sequence[RetrievedChunk],
        calibration: CalibrationResult,
    ) -> EvidenceDecision:
        identity = calibration.identity
        for field, actual, expected in (
            ("provider_name", self.provider.provider_name, identity.provider_name),
            ("model_name", self.provider.model_name, identity.model_name),
            ("dimension", self.provider.dimension, identity.dimension),
        ):
            if actual != expected:
                raise CalibrationMismatchError(
                    f"calibration {field} is {expected!r}, runtime is {actual!r}"
                )
        chunks = tuple(retrieved_chunks)
        if any(chunk.course_id != course_id for chunk in chunks):
            return EvidenceDecision(
                "abstain",
                "foreign_course_evidence",
                chunks[0].score if chunks else None,
                calibration.threshold,
                (),
            )
        if not chunks:
            return EvidenceDecision(
                "abstain",
                "no_evidence_retrieved",
                None,
                calibration.threshold,
                (),
            )
        if chunks[0].score < calibration.threshold:
            return EvidenceDecision(
                "abstain",
                "insufficient_retrieval_evidence",
                chunks[0].score,
                calibration.threshold,
                (),
            )
        return EvidenceDecision(
            "allow",
            "top1_score_meets_calibrated_threshold",
            chunks[0].score,
            calibration.threshold,
            chunks,
        )
