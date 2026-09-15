import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

from course_rag_api.models import RetrievedChunk


@dataclass(frozen=True, slots=True)
class ScoreFeatures:
    top1_score: float | None
    top2_score: float | None
    top3_score: float | None
    top1_top2_margin: float | None
    result_count: int


@dataclass(frozen=True, slots=True)
class LabeledScore:
    label: str
    answerable: bool
    top1_score: float | None
    top1_top2_margin: float | None


@dataclass(frozen=True, slots=True)
class CalibrationIdentity:
    provider_name: str
    model_name: str
    dimension: int
    dataset_version: str
    retrieval_limit: int
    chunk_target_size: int
    chunk_overlap: int


@dataclass(frozen=True, slots=True)
class GateMetrics:
    answerable_allow: int
    answerable_abstain: int
    unsupported_allow: int
    unsupported_abstain: int
    answerable_acceptance_rate: float
    unsupported_false_accept_rate: float
    abstention_rate: float


@dataclass(frozen=True, slots=True)
class DistributionStats:
    count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    median: float | None
    p10: float | None
    p25: float | None
    p75: float | None
    p90: float | None


@dataclass(frozen=True, slots=True)
class MarginAnalysis:
    margin_threshold: float | None
    metrics: GateMetrics | None
    candidate_count: int


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    identity: CalibrationIdentity
    policy_type: Literal["top1_threshold"]
    threshold: float
    calibration_metrics: GateMetrics
    candidate_count: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, path: Path) -> "CalibrationResult":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            identity = CalibrationIdentity(**raw["identity"])
            metrics = GateMetrics(**raw["calibration_metrics"])
            result = cls(
                identity=identity,
                policy_type=raw["policy_type"],
                threshold=float(raw["threshold"]),
                calibration_metrics=metrics,
                candidate_count=int(raw["candidate_count"]),
            )
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise ValueError(f"invalid calibration artifact: {path}") from error
        if result.policy_type != "top1_threshold" or not math.isfinite(
            result.threshold
        ):
            raise ValueError(f"invalid calibration artifact: {path}")
        return result


def extract_score_features(chunks: Sequence[RetrievedChunk]) -> ScoreFeatures:
    scores = [chunk.score for chunk in chunks]
    return ScoreFeatures(
        top1_score=scores[0] if scores else None,
        top2_score=scores[1] if len(scores) > 1 else None,
        top3_score=scores[2] if len(scores) > 2 else None,
        top1_top2_margin=scores[0] - scores[1] if len(scores) > 1 else None,
        result_count=len(scores),
    )


def summarize_distribution(values: Sequence[float]) -> DistributionStats:
    ordered = sorted(values)
    if not ordered:
        return DistributionStats(0, None, None, None, None, None, None, None, None)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return DistributionStats(
        count=len(ordered),
        minimum=ordered[0],
        maximum=ordered[-1],
        mean=statistics.fmean(ordered),
        median=statistics.median(ordered),
        p10=percentile(0.10),
        p25=percentile(0.25),
        p75=percentile(0.75),
        p90=percentile(0.90),
    )


def evaluate_policy(
    cases: Sequence[LabeledScore],
    *,
    threshold: float,
    margin_threshold: float | None = None,
) -> GateMetrics:
    decisions = [
        case.top1_score is not None
        and case.top1_score >= threshold
        and (
            margin_threshold is None
            or (
                case.top1_top2_margin is not None
                and case.top1_top2_margin >= margin_threshold
            )
        )
        for case in cases
    ]
    answerable_allow = sum(
        case.answerable and allowed
        for case, allowed in zip(cases, decisions, strict=True)
    )
    answerable_count = sum(case.answerable for case in cases)
    unsupported_allow = sum(
        not case.answerable and allowed
        for case, allowed in zip(cases, decisions, strict=True)
    )
    unsupported_count = len(cases) - answerable_count
    abstained = len(cases) - sum(decisions)
    return GateMetrics(
        answerable_allow=answerable_allow,
        answerable_abstain=answerable_count - answerable_allow,
        unsupported_allow=unsupported_allow,
        unsupported_abstain=unsupported_count - unsupported_allow,
        answerable_acceptance_rate=(
            answerable_allow / answerable_count if answerable_count else 0.0
        ),
        unsupported_false_accept_rate=(
            unsupported_allow / unsupported_count if unsupported_count else 0.0
        ),
        abstention_rate=abstained / len(cases) if cases else 0.0,
    )


def calibrate_threshold(
    cases: Sequence[LabeledScore], identity: CalibrationIdentity
) -> CalibrationResult:
    if not cases or not any(case.answerable for case in cases) or not any(
        not case.answerable for case in cases
    ):
        raise ValueError("calibration requires answerable and unsupported cases")
    scores = sorted(
        {case.top1_score for case in cases if case.top1_score is not None}
    )
    if not scores:
        raise ValueError("calibration requires at least one retrieval score")
    candidates = (*scores, math.nextafter(scores[-1], math.inf))
    evaluated = [
        (threshold, evaluate_policy(cases, threshold=threshold))
        for threshold in candidates
    ]
    threshold, metrics = min(
        evaluated,
        key=lambda item: (
            item[1].unsupported_allow,
            -item[1].answerable_allow,
            -item[0],
        ),
    )
    return CalibrationResult(
        identity=identity,
        policy_type="top1_threshold",
        threshold=threshold,
        calibration_metrics=metrics,
        candidate_count=len(candidates),
    )


def calibrate_margin_threshold(
    cases: Sequence[LabeledScore], *, threshold: float
) -> MarginAnalysis:
    margins = sorted(
        {
            case.top1_top2_margin
            for case in cases
            if case.top1_score is not None
            and case.top1_score >= threshold
            and case.top1_top2_margin is not None
        }
    )
    if not margins:
        return MarginAnalysis(None, None, 0)
    candidates = (*margins, math.nextafter(margins[-1], math.inf))
    margin_threshold, metrics = min(
        (
            (
                candidate,
                evaluate_policy(
                    cases, threshold=threshold, margin_threshold=candidate
                ),
            )
            for candidate in candidates
        ),
        key=lambda item: (
            item[1].unsupported_allow,
            -item[1].answerable_allow,
            -item[0],
        ),
    )
    return MarginAnalysis(margin_threshold, metrics, len(candidates))
