import json
from pathlib import Path

import pytest

from course_rag_api.embeddings import DeterministicEmbeddingProvider
from course_rag_api.evaluation import (
    EvaluationCase,
    EvidenceRef,
    SupportConflictCase,
    load_evaluation_dataset,
    run_retrieval_evaluation,
    score_case,
    summarize_support_verification,
    summarize_results,
)
from course_rag_api.models import RetrievedChunk
from course_rag_api.support import SupportDecision, SupportStatus


DATASET = Path(__file__).parents[2] / "examples/retrieval-evaluation/dataset.json"


def _chunk(
    *, course_id: str, filename: str, chunk_index: int, score: float
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{course_id}-{filename}-{chunk_index}",
        course_id=course_id,
        document_id=f"document-{filename}",
        filename=filename,
        text="synthetic evidence",
        score=score,
        chunk_index=chunk_index,
        source_type="section",
        source_start=chunk_index + 1,
        source_end=chunk_index + 1,
    )


def test_dataset_validation_rejects_answerable_case_without_expected_evidence(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text("Synthetic evidence.")
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps(
            {
                "version": "test-v1",
                "chunk_target_size": 40,
                "chunk_overlap": 5,
                "courses": [
                    {"key": "course-a", "name": "Course A", "documents": ["a.txt"]}
                ],
                "cases": [
                    {
                        "label": "missing-label",
                        "category": "exact-answer",
                        "course": "course-a",
                        "query": "Where is the answer?",
                        "answerable": True,
                        "expected_evidence": [],
                        "forbidden_courses": [],
                        "split": "calibration",
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="expected_evidence"):
        load_evaluation_dataset(path)


def test_hit_recall_and_score_metrics_are_deterministic() -> None:
    case = EvaluationCase(
        label="two-facts",
        category="same-course cross-document",
        course="orbital",
        query="Which two facts apply?",
        answerable=True,
        expected_evidence=(
            EvidenceRef("alpha.md", 0, "section", 1, 1),
            EvidenceRef("beta.md", 0, "section", 1, 1),
        ),
        forbidden_courses=("marine",),
    )
    results = (
        _chunk(course_id="orbital-id", filename="alpha.md", chunk_index=0, score=0.8),
        _chunk(course_id="orbital-id", filename="noise.md", chunk_index=0, score=0.4),
        _chunk(course_id="orbital-id", filename="beta.md", chunk_index=0, score=0.3),
    )

    scored = score_case(
        case,
        results,
        course_keys_by_id={"orbital-id": "orbital", "marine-id": "marine"},
    )
    summary = summarize_results((scored,))

    assert scored.top_1_score == 0.8
    assert scored.scores == (0.8, 0.4, 0.3)
    assert scored.hit_at == {1: True, 3: True, 5: True}
    assert scored.recall_at == {1: 0.5, 3: 1.0, 5: 1.0}
    assert scored.isolation_passed is True
    assert summary.hit_at == {1: 1.0, 3: 1.0, 5: 1.0}
    assert summary.recall_at == {1: 0.5, 3: 1.0, 5: 1.0}


def test_unsupported_case_collects_scores_but_never_counts_as_a_hit() -> None:
    case = EvaluationCase(
        label="unsupported",
        category="unsupported query",
        course="orbital",
        query="What is not in the corpus?",
        answerable=False,
        expected_evidence=(),
        forbidden_courses=("marine",),
    )
    scored = score_case(
        case,
        (
            _chunk(
                course_id="orbital-id",
                filename="related.md",
                chunk_index=0,
                score=0.6,
            ),
        ),
        course_keys_by_id={"orbital-id": "orbital", "marine-id": "marine"},
    )

    assert scored.top_1_score == 0.6
    assert scored.scores == (0.6,)
    assert scored.hit_at == {1: False, 3: False, 5: False}
    assert scored.recall_at == {1: None, 3: None, 5: None}
    assert summarize_results((scored,)).unsupported_count == 1


def test_forbidden_course_result_fails_isolation() -> None:
    case = EvaluationCase(
        label="contradiction",
        category="contradictory-course isolation",
        course="orbital",
        query="Which coefficient applies?",
        answerable=True,
        expected_evidence=(EvidenceRef("orbital.md", 0, "section", 1, 1),),
        forbidden_courses=("marine",),
    )

    scored = score_case(
        case,
        (
            _chunk(
                course_id="marine-id",
                filename="orbital.md",
                chunk_index=0,
                score=0.9,
            ),
        ),
        course_keys_by_id={"orbital-id": "orbital", "marine-id": "marine"},
    )

    assert scored.isolation_passed is False
    assert scored.hit_at == {1: False, 3: False, 5: False}
    assert scored.recall_at == {1: 0.0, 3: 0.0, 5: 0.0}


def test_synthetic_dataset_runs_offline_and_never_returns_forbidden_course() -> None:
    dataset = load_evaluation_dataset(DATASET)
    first = run_retrieval_evaluation(
        dataset, DeterministicEmbeddingProvider(dimension=256), limit=5
    )
    second = run_retrieval_evaluation(
        dataset, DeterministicEmbeddingProvider(dimension=256), limit=5
    )

    assert first.to_json() == second.to_json()
    assert dataset.version == "m2b-v1"
    assert len(dataset.cases) == 40
    calibration_labels = {
        case.label for case in dataset.cases if case.split == "calibration"
    }
    holdout_labels = {case.label for case in dataset.cases if case.split == "holdout"}
    assert len(calibration_labels) == 28
    assert len(holdout_labels) == 12
    assert calibration_labels.isdisjoint(holdout_labels)
    assert dataset.support_conflicts[0].label == "helios-pressure-conflict"
    assert all(case.answerable == bool(case.expected_evidence) for case in dataset.cases)
    assert first.case_count == len(dataset.cases)
    assert first.answerable_count > 0
    assert first.unsupported_count > 0
    assert all(case.isolation_passed for case in first.cases)
    assert all(
        evidence.course not in dataset_case.forbidden_courses
        for result, dataset_case in zip(first.cases, dataset.cases, strict=True)
        for evidence in result.retrieved
    )
    assert {
        "exact-answer",
        "paraphrase",
        "same-course cross-document",
        "multi-chunk",
        "unsupported unrelated",
        "unsupported plausible domain",
        "wrong-attribute",
        "adversarial lexical overlap",
        "near-miss",
        "contradictory-course isolation",
        "vocabulary-overlap isolation",
    } <= {case.category for case in dataset.cases}


def test_support_verification_metrics_remain_separate_from_retrieval_metrics() -> None:
    answerable = EvaluationCase(
        label="answerable",
        category="exact-answer",
        course="orbital",
        query="What is the coefficient?",
        answerable=True,
        expected_evidence=(EvidenceRef("alpha.md", 0, "section", 1, 1),),
        forbidden_courses=("marine",),
    )
    unsupported = EvaluationCase(
        label="unsupported",
        category="wrong-attribute",
        course="orbital",
        query="Who discovered the coefficient?",
        answerable=False,
        expected_evidence=(),
        forbidden_courses=("marine",),
    )
    decisions = {
        "answerable": SupportDecision(
            SupportStatus.SUPPORTED,
            "explicit_support",
            ("chunk-a",),
            "scripted",
            "test-v1",
        ),
        "unsupported": SupportDecision(
            SupportStatus.INSUFFICIENT,
            "missing_attribute",
            (),
            "scripted",
            "test-v1",
        ),
    }

    conflict = SupportConflictCase(
        "helios-pressure-conflict",
        "orbital",
        "What is the Helios pressure?",
        ("The Helios pressure is 81 kPa.", "The Helios pressure is 93 kPa."),
        SupportStatus.CONFLICTING,
    )
    metrics = summarize_support_verification(
        (answerable, unsupported),
        decisions,
        conflict_cases=(conflict,),
        conflict_decisions={
            conflict.label: SupportDecision(
                SupportStatus.CONFLICTING,
                "incompatible_candidate_answers",
                ("chunk-a", "chunk-b"),
                "scripted",
                "test-v1",
            )
        },
    )

    assert metrics.supported_answerable == 1
    assert metrics.missed_answerable == 0
    assert metrics.incorrectly_supported_unsupported == 0
    assert metrics.correctly_rejected_unsupported == 1
    assert metrics.correctly_detected_conflicts == 1
    assert metrics.answerable_support_rate == 1.0
    assert metrics.unsupported_false_support_rate == 0.0
    assert metrics.conflict_detection_rate == 1.0
