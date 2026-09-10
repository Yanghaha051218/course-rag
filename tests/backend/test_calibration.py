import pytest

from course_rag_api.calibration import (
    CalibrationIdentity,
    LabeledScore,
    calibrate_threshold,
    evaluate_policy,
    extract_score_features,
    summarize_distribution,
)
from course_rag_api.models import RetrievedChunk


def _chunk(score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{score}",
        course_id="course-a",
        document_id="document-a",
        filename="lesson.md",
        text="Synthetic evidence.",
        score=score,
        chunk_index=0,
        source_type="section",
        source_start=1,
        source_end=1,
    )


def _identity() -> CalibrationIdentity:
    return CalibrationIdentity(
        provider_name="deterministic",
        model_name="deterministic-hash-v1",
        dimension=256,
        dataset_version="m2b-v1",
        retrieval_limit=5,
        chunk_target_size=32,
        chunk_overlap=4,
    )


def test_score_features_handle_three_one_and_zero_results() -> None:
    full = extract_score_features((_chunk(0.8), _chunk(0.5), _chunk(0.2)))
    single = extract_score_features((_chunk(0.4),))
    empty = extract_score_features(())

    assert (full.top1_score, full.top2_score, full.top3_score) == (0.8, 0.5, 0.2)
    assert full.top1_top2_margin == pytest.approx(0.3)
    assert full.result_count == 3
    assert single.top2_score is None
    assert single.top1_top2_margin is None
    assert empty.top1_score is None
    assert empty.result_count == 0


def test_calibration_prioritizes_false_accepts_then_answerable_coverage() -> None:
    cases = (
        LabeledScore("answer-high", True, 0.9, 0.3),
        LabeledScore("answer-mid", True, 0.7, 0.2),
        LabeledScore("answer-low", True, 0.5, 0.1),
        LabeledScore("unsupported-high", False, 0.6, 0.2),
        LabeledScore("unsupported-low", False, 0.4, 0.1),
    )

    first = calibrate_threshold(cases, _identity())
    second = calibrate_threshold(cases, _identity())

    assert first == second
    assert first.threshold == 0.7
    assert first.policy_type == "top1_threshold"
    assert first.calibration_metrics.unsupported_allow == 0
    assert first.calibration_metrics.answerable_allow == 2
    assert first.calibration_metrics.answerable_abstain == 1


def test_gate_metrics_report_confusion_and_rates() -> None:
    cases = (
        LabeledScore("answer-allow", True, 0.8, 0.2),
        LabeledScore("answer-abstain", True, 0.4, 0.1),
        LabeledScore("unsupported-allow", False, 0.6, 0.2),
        LabeledScore("unsupported-abstain", False, 0.2, 0.1),
    )

    metrics = evaluate_policy(cases, threshold=0.5)

    assert metrics.answerable_allow == 1
    assert metrics.answerable_abstain == 1
    assert metrics.unsupported_allow == 1
    assert metrics.unsupported_abstain == 1
    assert metrics.answerable_acceptance_rate == 0.5
    assert metrics.unsupported_false_accept_rate == 0.5
    assert metrics.abstention_rate == 0.5


def test_distribution_summary_reports_stable_percentiles() -> None:
    summary = summarize_distribution((0.1, 0.2, 0.3, 0.4, 0.5))

    assert summary.count == 5
    assert summary.minimum == 0.1
    assert summary.maximum == 0.5
    assert summary.mean == pytest.approx(0.3)
    assert summary.median == 0.3
    assert summary.p10 == pytest.approx(0.14)
    assert summary.p90 == pytest.approx(0.46)
