import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from qdrant_client import QdrantClient

from course_rag_api.calibration import extract_score_features
from course_rag_api.embeddings import EmbeddingProvider
from course_rag_api.indexing import IndexingService
from course_rag_api.ingestion import ingest_document
from course_rag_api.models import RetrievedChunk, SourceType
from course_rag_api.retrieval import Retriever
from course_rag_api.storage import SQLiteStore
from course_rag_api.support import SupportDecision, SupportStatus
from course_rag_api.vector_store import QdrantVectorIndex


_K_VALUES = (1, 3, 5)


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    filename: str
    chunk_index: int
    source_type: SourceType
    source_start: int
    source_end: int

    def matches(self, chunk: RetrievedChunk) -> bool:
        return (
            self.filename,
            self.chunk_index,
            self.source_type,
            self.source_start,
            self.source_end,
        ) == (
            chunk.filename,
            chunk.chunk_index,
            chunk.source_type,
            chunk.source_start,
            chunk.source_end,
        )


@dataclass(frozen=True, slots=True)
class EvaluationCourse:
    key: str
    name: str
    documents: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    label: str
    category: str
    course: str
    query: str
    answerable: bool
    expected_evidence: tuple[EvidenceRef, ...]
    forbidden_courses: tuple[str, ...]
    split: Literal["calibration", "holdout"] = "calibration"


@dataclass(frozen=True, slots=True)
class SupportConflictCase:
    label: str
    course: str
    query: str
    evidence: tuple[str, ...]
    expected_status: SupportStatus


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    version: str
    chunk_target_size: int
    chunk_overlap: int
    courses: tuple[EvaluationCourse, ...]
    cases: tuple[EvaluationCase, ...]
    support_conflicts: tuple[SupportConflictCase, ...]


@dataclass(frozen=True, slots=True)
class RetrievedEvidence:
    rank: int
    course: str
    filename: str
    chunk_index: int
    source_type: SourceType
    source_start: int
    source_end: int
    score: float


@dataclass(frozen=True, slots=True)
class EvaluationCaseResult:
    label: str
    category: str
    course: str
    course_id: str
    split: Literal["calibration", "holdout"]
    answerable: bool
    expected_evidence: tuple[EvidenceRef, ...]
    top_1_score: float | None
    top_2_score: float | None
    top_3_score: float | None
    top_1_top_2_margin: float | None
    retrieved_result_count: int
    scores: tuple[float, ...]
    hit_at: dict[int, bool]
    recall_at: dict[int, float | None]
    isolation_passed: bool
    retrieved: tuple[RetrievedEvidence, ...]


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    case_count: int
    answerable_count: int
    unsupported_count: int
    hit_at: dict[int, float]
    recall_at: dict[int, float]
    isolation_passed: bool
    cases: tuple[EvaluationCaseResult, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class SupportVerificationMetrics:
    supported_answerable: int
    missed_answerable: int
    incorrectly_supported_unsupported: int
    correctly_rejected_unsupported: int
    correctly_detected_conflicts: int
    answerable_support_rate: float
    unsupported_false_support_rate: float
    conflict_detection_rate: float


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def load_evaluation_dataset(path: Path) -> EvaluationDataset:
    dataset_path = Path(path).resolve()
    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read evaluation dataset: {dataset_path}") from error
    if not isinstance(raw, dict):
        raise ValueError("evaluation dataset must be a JSON object")

    version = _nonempty_string(raw.get("version"), "version")
    target_size = _positive_int(raw.get("chunk_target_size"), "chunk_target_size")
    overlap = raw.get("chunk_overlap")
    if (
        isinstance(overlap, bool)
        or not isinstance(overlap, int)
        or not 0 <= overlap < target_size
    ):
        raise ValueError(
            "chunk_overlap must be a non-negative integer below chunk_target_size"
        )

    raw_courses = raw.get("courses")
    if not isinstance(raw_courses, list) or not raw_courses:
        raise ValueError("courses must be a non-empty list")
    courses: list[EvaluationCourse] = []
    course_documents: dict[str, set[str]] = {}
    for index, item in enumerate(raw_courses):
        if not isinstance(item, dict):
            raise ValueError(f"courses[{index}] must be an object")
        key = _nonempty_string(item.get("key"), f"courses[{index}].key")
        if key in course_documents:
            raise ValueError(f"duplicate course key: {key}")
        name = _nonempty_string(item.get("name"), f"courses[{index}].name")
        raw_documents = item.get("documents")
        if not isinstance(raw_documents, list) or not raw_documents:
            raise ValueError(f"courses[{index}].documents must be a non-empty list")
        documents: list[Path] = []
        filenames: set[str] = set()
        for document_index, value in enumerate(raw_documents):
            relative = _nonempty_string(
                value, f"courses[{index}].documents[{document_index}]"
            )
            document = (dataset_path.parent / relative).resolve()
            if not document.is_file():
                raise ValueError(f"evaluation document does not exist: {relative}")
            if document.name in filenames:
                raise ValueError(f"duplicate filename in course {key}: {document.name}")
            documents.append(document)
            filenames.add(document.name)
        courses.append(EvaluationCourse(key, name, tuple(documents)))
        course_documents[key] = filenames

    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("cases must be a non-empty list")
    cases: list[EvaluationCase] = []
    labels: set[str] = set()
    for index, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise ValueError(f"cases[{index}] must be an object")
        label = _nonempty_string(item.get("label"), f"cases[{index}].label")
        if label in labels:
            raise ValueError(f"duplicate case label: {label}")
        labels.add(label)
        category = _nonempty_string(item.get("category"), f"cases[{index}].category")
        course = _nonempty_string(item.get("course"), f"cases[{index}].course")
        if course not in course_documents:
            raise ValueError(f"unknown course in case {label}: {course}")
        query = _nonempty_string(item.get("query"), f"cases[{index}].query")
        answerable = item.get("answerable")
        if not isinstance(answerable, bool):
            raise ValueError(f"answerable must be boolean in case {label}")
        raw_evidence = item.get("expected_evidence")
        if not isinstance(raw_evidence, list):
            raise ValueError(f"expected_evidence must be a list in case {label}")
        evidence: list[EvidenceRef] = []
        for evidence_index, reference in enumerate(raw_evidence):
            if not isinstance(reference, dict):
                raise ValueError(
                    f"expected_evidence[{evidence_index}] must be an object "
                    f"in case {label}"
                )
            filename = _nonempty_string(
                reference.get("filename"), "expected_evidence.filename"
            )
            if filename not in course_documents[course]:
                raise ValueError(
                    f"unknown evidence filename in case {label}: {filename}"
                )
            source_type = reference.get("source_type")
            if source_type not in ("page", "slide", "section"):
                raise ValueError(f"invalid source_type in case {label}")
            source_start = _positive_int(reference.get("source_start"), "source_start")
            source_end = _positive_int(reference.get("source_end"), "source_end")
            if source_end < source_start:
                raise ValueError(f"source_end precedes source_start in case {label}")
            evidence.append(
                EvidenceRef(
                    filename,
                    _nonnegative_int(reference.get("chunk_index"), "chunk_index"),
                    source_type,
                    source_start,
                    source_end,
                )
            )
        if answerable != bool(evidence):
            raise ValueError(
                f"expected_evidence must be non-empty only for answerable case {label}"
            )
        raw_forbidden = item.get("forbidden_courses", [])
        if not isinstance(raw_forbidden, list):
            raise ValueError(f"forbidden_courses must be a list in case {label}")
        forbidden = tuple(
            _nonempty_string(value, f"forbidden_courses in case {label}")
            for value in raw_forbidden
        )
        if course in forbidden or any(
            value not in course_documents for value in forbidden
        ):
            raise ValueError(f"invalid forbidden course in case {label}")
        split = item.get("split")
        if split not in ("calibration", "holdout"):
            raise ValueError(f"invalid split in case {label}")
        cases.append(
            EvaluationCase(
                label,
                category,
                course,
                query,
                answerable,
                tuple(evidence),
                forbidden,
                split,
            )
        )

    raw_conflicts = raw.get("support_conflicts", [])
    if not isinstance(raw_conflicts, list):
        raise ValueError("support_conflicts must be a list")
    conflicts: list[SupportConflictCase] = []
    conflict_labels: set[str] = set()
    for index, item in enumerate(raw_conflicts):
        if not isinstance(item, dict):
            raise ValueError(f"support_conflicts[{index}] must be an object")
        label = _nonempty_string(item.get("label"), f"support_conflicts[{index}].label")
        if label in conflict_labels or label in labels:
            raise ValueError(f"duplicate support conflict label: {label}")
        conflict_labels.add(label)
        course = _nonempty_string(
            item.get("course"), f"support_conflicts[{index}].course"
        )
        if course not in course_documents:
            raise ValueError(f"unknown course in support conflict {label}: {course}")
        query = _nonempty_string(item.get("query"), f"support_conflicts[{index}].query")
        evidence = item.get("evidence")
        if (
            not isinstance(evidence, list)
            or len(evidence) < 2
            or any(not isinstance(text, str) or not text.strip() for text in evidence)
        ):
            raise ValueError(
                f"support conflict {label} requires two non-empty evidence texts"
            )
        if item.get("expected_status") != SupportStatus.CONFLICTING.value:
            raise ValueError(f"support conflict {label} must expect CONFLICTING")
        conflicts.append(
            SupportConflictCase(
                label, course, query, tuple(evidence), SupportStatus.CONFLICTING
            )
        )

    return EvaluationDataset(
        version, target_size, overlap, tuple(courses), tuple(cases), tuple(conflicts)
    )


def score_case(
    case: EvaluationCase,
    chunks: tuple[RetrievedChunk, ...],
    *,
    course_keys_by_id: dict[str, str],
) -> EvaluationCaseResult:
    retrieved: list[RetrievedEvidence] = []
    for rank, chunk in enumerate(chunks, start=1):
        try:
            course = course_keys_by_id[chunk.course_id]
        except KeyError as error:
            raise ValueError(f"retrieved unknown course: {chunk.course_id}") from error
        retrieved.append(
            RetrievedEvidence(
                rank,
                course,
                chunk.filename,
                chunk.chunk_index,
                chunk.source_type,
                chunk.source_start,
                chunk.source_end,
                chunk.score,
            )
        )
    relevant_ranks = {
        rank
        for rank, chunk in enumerate(chunks, start=1)
        if course_keys_by_id[chunk.course_id] == case.course
        and any(reference.matches(chunk) for reference in case.expected_evidence)
    }
    hit_at = {
        k: case.answerable and any(rank <= k for rank in relevant_ranks)
        for k in _K_VALUES
    }
    recall_at = {
        k: (
            len(
                {
                    reference
                    for reference in case.expected_evidence
                    if any(
                        course_keys_by_id[chunk.course_id] == case.course
                        and reference.matches(chunk)
                        for chunk in chunks[:k]
                    )
                }
            )
            / len(case.expected_evidence)
            if case.answerable
            else None
        )
        for k in _K_VALUES
    }
    features = extract_score_features(chunks)
    return EvaluationCaseResult(
        label=case.label,
        category=case.category,
        course=case.course,
        course_id=case.course,
        split=case.split,
        answerable=case.answerable,
        expected_evidence=case.expected_evidence,
        top_1_score=features.top1_score,
        top_2_score=features.top2_score,
        top_3_score=features.top3_score,
        top_1_top_2_margin=features.top1_top2_margin,
        retrieved_result_count=features.result_count,
        scores=tuple(chunk.score for chunk in chunks),
        hit_at=hit_at,
        recall_at=recall_at,
        isolation_passed=all(
            item.course == case.course and item.course not in case.forbidden_courses
            for item in retrieved
        ),
        retrieved=tuple(retrieved),
    )


def summarize_results(results: tuple[EvaluationCaseResult, ...]) -> EvaluationReport:
    answerable = tuple(result for result in results if result.answerable)
    denominator = len(answerable)
    hit_at = {
        k: sum(result.hit_at[k] for result in answerable) / denominator
        if denominator
        else 0.0
        for k in _K_VALUES
    }
    recall_at = {
        k: sum(float(result.recall_at[k]) for result in answerable) / denominator
        if denominator
        else 0.0
        for k in _K_VALUES
    }
    return EvaluationReport(
        len(results),
        denominator,
        len(results) - denominator,
        hit_at,
        recall_at,
        all(result.isolation_passed for result in results),
        results,
    )


def summarize_support_verification(
    cases: Sequence[EvaluationCase],
    decisions: Mapping[str, SupportDecision],
    *,
    conflict_cases: Sequence[SupportConflictCase] = (),
    conflict_decisions: Mapping[str, SupportDecision] | None = None,
) -> SupportVerificationMetrics:
    if {case.label for case in cases} != set(decisions):
        raise ValueError(
            "support decisions must cover each evaluation case exactly once"
        )
    answerable = [case for case in cases if case.answerable]
    unsupported = [case for case in cases if not case.answerable]
    supported_answerable = sum(
        decisions[case.label].status is SupportStatus.SUPPORTED for case in answerable
    )
    incorrectly_supported_unsupported = sum(
        decisions[case.label].status is SupportStatus.SUPPORTED for case in unsupported
    )
    conflict_decisions = conflict_decisions or {}
    if {case.label for case in conflict_cases} != set(conflict_decisions):
        raise ValueError("support decisions must cover each conflict case exactly once")
    conflict_count = len(conflict_cases)
    correctly_detected_conflicts = sum(
        conflict_decisions[case.label].status is case.expected_status
        for case in conflict_cases
    )
    return SupportVerificationMetrics(
        supported_answerable=supported_answerable,
        missed_answerable=len(answerable) - supported_answerable,
        incorrectly_supported_unsupported=incorrectly_supported_unsupported,
        correctly_rejected_unsupported=(
            len(unsupported) - incorrectly_supported_unsupported
        ),
        correctly_detected_conflicts=correctly_detected_conflicts,
        answerable_support_rate=(
            supported_answerable / len(answerable) if answerable else 0.0
        ),
        unsupported_false_support_rate=(
            incorrectly_supported_unsupported / len(unsupported)
            if unsupported
            else 0.0
        ),
        conflict_detection_rate=(
            correctly_detected_conflicts / conflict_count if conflict_count else 0.0
        ),
    )


def format_evaluation_report(report: EvaluationReport) -> str:
    lines = [
        "Retrieval evaluation",
        (
            f"Cases: {report.case_count} "
            f"(answerable={report.answerable_count}, "
            f"unsupported={report.unsupported_count})"
        ),
    ]
    for k in _K_VALUES:
        hits = round(report.hit_at[k] * report.answerable_count)
        lines.append(
            f"Hit@{k}: {hits}/{report.answerable_count} ({report.hit_at[k]:.3f})"
        )
        lines.append(f"Recall@{k}: {report.recall_at[k]:.3f}")
    lines.append(f"Isolation: {'PASS' if report.isolation_passed else 'FAIL'}")
    unsupported = [case for case in report.cases if not case.answerable]
    if unsupported:
        lines.append("Unsupported top-1 scores:")
        lines.extend(
            f"- {case.label}: "
            + ("none" if case.top_1_score is None else f"{case.top_1_score:.6f}")
            for case in unsupported
        )
    return "\n".join(lines)


def run_retrieval_evaluation(
    dataset: EvaluationDataset, provider: EmbeddingProvider, *, limit: int = 5
) -> EvaluationReport:
    if not 1 <= limit <= 100:
        raise ValueError("evaluation limit must be between 1 and 100")
    with TemporaryDirectory(prefix="course-rag-evaluation-") as directory:
        store = SQLiteStore(Path(directory) / "evaluation.sqlite3")
        client = QdrantClient(location=":memory:")
        try:
            index = QdrantVectorIndex(client, "course_rag_evaluation", provider)
            indexing = IndexingService(store, index, provider)
            retriever = Retriever(store, index, provider)
            course_ids: dict[str, str] = {}
            for course in dataset.courses:
                course_id = store.create_course(course.name).id
                course_ids[course.key] = course_id
                for document in course.documents:
                    ingest_document(
                        store=store,
                        course_id=course_id,
                        file_path=document,
                        chunk_target_size=dataset.chunk_target_size,
                        chunk_overlap=dataset.chunk_overlap,
                    )
                indexing.index_course(course_id)
            course_keys_by_id = {value: key for key, value in course_ids.items()}
            results: list[EvaluationCaseResult] = []
            for case in dataset.cases:
                chunks = retriever.retrieve(
                    course_id=course_ids[case.course], query=case.query, limit=100
                )
                stable_chunks = tuple(
                    sorted(
                        chunks,
                        key=lambda chunk: (
                            -chunk.score,
                            chunk.filename,
                            chunk.chunk_index,
                            chunk.source_start,
                            chunk.source_end,
                        ),
                    )[:limit]
                )
                results.append(
                    score_case(
                        case,
                        stable_chunks,
                        course_keys_by_id=course_keys_by_id,
                    )
                )
            return summarize_results(tuple(results))
        finally:
            client.close()
