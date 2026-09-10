import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

from course_rag_api.generation import (
    AnswerGenerationService,
    FinalResponse,
    FinalStatus,
    Generator,
)
from course_rag_api.models import RetrievedChunk, SourceType
from course_rag_api.retrieval import Retriever
from course_rag_api.support import SupportStatus, SupportVerificationService


_K_VALUES = (1, 3, 5)
_CASE_TYPES = {"answerable", "unsupported", "conflict"}
_CATEGORIES = {
    "direct_factual",
    "concept_explanation",
    "multi_document",
    "near_miss",
    "citation_sensitive",
}


@dataclass(frozen=True, slots=True)
class BenchmarkSource:
    document: str
    source_type: SourceType
    start: int
    end: int

    def matches(self, chunk: RetrievedChunk) -> bool:
        return (
            self.document,
            self.source_type,
            self.start,
            self.end,
        ) == (
            chunk.filename,
            chunk.source_type,
            chunk.source_start,
            chunk.source_end,
        )


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    id: str
    course_id: str
    question: str
    type: Literal["answerable", "unsupported", "conflict"]
    category: str
    expected_sources: tuple[BenchmarkSource, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkDataset:
    version: str
    cases: tuple[BenchmarkCase, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkFailure:
    category: Literal[
        "retrieval_failure",
        "support_verification_failure",
        "citation_failure",
        "generation_failure",
        "dataset_error",
    ]
    detail: str


@dataclass(frozen=True, slots=True)
class RetrievedSource:
    rank: int
    chunk_id: str
    document: str
    source_type: SourceType
    start: int
    end: int
    score: float


@dataclass(frozen=True, slots=True)
class BenchmarkCaseResult:
    case_id: str
    course_id: str
    question: str
    type: str
    category: str
    retrieval_hit_at: dict[int, bool]
    retrieval_recall_at: dict[int, float | None]
    reciprocal_rank: float | None
    retrieved: tuple[RetrievedSource, ...]
    support_status: str | None
    final_status: str | None
    citation_valid: bool | None
    failures: tuple[BenchmarkFailure, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    retrieval_hit_at: dict[int, float]
    retrieval_recall_at: dict[int, float]
    retrieval_mrr: float
    support_accept_rate: float | None
    false_support_rate: float | None
    conflict_detection_rate: float | None
    citation_validity_rate: float | None
    unsupported_claim_rate: float | None


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    dataset_version: str
    course_id: str
    embedding: dict[str, object]
    retrieval_k: int
    support_verifier: dict[str, str] | None
    generator: dict[str, str] | None
    metrics: BenchmarkMetrics
    cases: tuple[BenchmarkCaseResult, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def format_benchmark_report(report: BenchmarkReport) -> str:
    lines = [
        "Benchmark evaluation",
        f"Dataset: {report.dataset_version}",
        f"Course: {report.course_id}",
    ]
    for k in _K_VALUES:
        lines.append(f"Hit@{k}: {report.metrics.retrieval_hit_at[k]:.3f}")
        lines.append(f"Recall@{k}: {report.metrics.retrieval_recall_at[k]:.3f}")
    lines.append(f"MRR: {report.metrics.retrieval_mrr:.3f}")
    for label, value in (
        ("Support accept", report.metrics.support_accept_rate),
        ("False support", report.metrics.false_support_rate),
        ("Conflict detection", report.metrics.conflict_detection_rate),
        ("Citation validity", report.metrics.citation_validity_rate),
        ("Unsupported claim", report.metrics.unsupported_claim_rate),
    ):
        lines.append(f"{label}: {'not run' if value is None else f'{value:.3f}'}")
    lines.append("Failure analysis")
    failures = [
        (case, failure) for case in report.cases for failure in case.failures
    ]
    if not failures:
        lines.append("- none")
    for case, failure in failures:
        lines.append(f"- {case.case_id}: {failure.category} — {failure.detail}")
    return "\n".join(lines)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _source(value: object, case_id: str) -> BenchmarkSource:
    if not isinstance(value, dict) or set(value) != {
        "document",
        "source_type",
        "start",
        "end",
    }:
        raise ValueError(f"invalid supporting source in case {case_id}")
    document = _string(value["document"], f"document in case {case_id}")
    if Path(document).name != document:
        raise ValueError(f"document in case {case_id} must be a filename")
    source_type = value["source_type"]
    if source_type not in ("page", "slide", "section"):
        raise ValueError(f"invalid source_type in case {case_id}")
    start = _positive_int(value["start"], f"start in case {case_id}")
    end = _positive_int(value["end"], f"end in case {case_id}")
    if end < start:
        raise ValueError(f"end precedes start in case {case_id}")
    return BenchmarkSource(document, source_type, start, end)


def load_benchmark_dataset(path: Path) -> BenchmarkDataset:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read benchmark dataset: {path}") from error
    if not isinstance(raw, dict) or set(raw) != {"version", "cases"}:
        raise ValueError("benchmark dataset must contain only version and cases")
    version = _string(raw["version"], "version")
    raw_cases = raw["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("cases must be a non-empty list")
    cases: list[BenchmarkCase] = []
    ids: set[str] = set()
    for item in raw_cases:
        if not isinstance(item, dict) or set(item) != {
            "id",
            "course_id",
            "question",
            "type",
            "category",
            "expected",
        }:
            raise ValueError("invalid benchmark case")
        case_id = _string(item["id"], "case id")
        if case_id in ids:
            raise ValueError(f"duplicate benchmark case id: {case_id}")
        ids.add(case_id)
        case_type = item["type"]
        if case_type not in _CASE_TYPES:
            raise ValueError("invalid benchmark case type")
        category = _string(item["category"], f"category in case {case_id}")
        if category not in _CATEGORIES:
            raise ValueError(f"invalid benchmark category in case {case_id}")
        expected = item["expected"]
        if not isinstance(expected, dict) or set(expected) != {"supporting_sources"}:
            raise ValueError(f"invalid expected evidence in case {case_id}")
        raw_sources = expected["supporting_sources"]
        if not isinstance(raw_sources, list):
            raise ValueError(f"supporting_sources must be a list in case {case_id}")
        sources = tuple(_source(source, case_id) for source in raw_sources)
        if len(set(sources)) != len(sources):
            raise ValueError(f"duplicate supporting source in case {case_id}")
        if case_type == "unsupported" and sources:
            raise ValueError(f"unsupported case {case_id} cannot contain supporting sources")
        if case_type == "answerable" and not sources:
            raise ValueError(f"answerable case {case_id} requires supporting sources")
        if case_type == "conflict" and len(sources) < 2:
            raise ValueError(f"conflict case {case_id} requires two supporting sources")
        cases.append(
            BenchmarkCase(
                case_id,
                _string(item["course_id"], f"course_id in case {case_id}"),
                _string(item["question"], f"question in case {case_id}"),
                case_type,
                category,
                sources,
            )
        )
    if len({case.course_id for case in cases}) != 1:
        raise ValueError("a benchmark dataset must contain one course_id")
    return BenchmarkDataset(version, tuple(cases))


def _retrieval_result(case: BenchmarkCase, chunks: Sequence[RetrievedChunk]) -> tuple[
    dict[int, bool], dict[int, float | None], float | None, tuple[RetrievedSource, ...]
]:
    ranks = {
        source: rank
        for source in case.expected_sources
        for rank, chunk in enumerate(chunks, start=1)
        if source.matches(chunk)
    }
    hit = {
        k: bool(case.expected_sources) and any(rank <= k for rank in ranks.values())
        for k in _K_VALUES
    }
    recall = {
        k: (
            sum(rank <= k for rank in ranks.values()) / len(case.expected_sources)
            if case.expected_sources
            else None
        )
        for k in _K_VALUES
    }
    reciprocal_rank = 1 / min(ranks.values()) if ranks else None
    retrieved = tuple(
        RetrievedSource(
            rank,
            chunk.chunk_id,
            chunk.filename,
            chunk.source_type,
            chunk.source_start,
            chunk.source_end,
            chunk.score,
        )
        for rank, chunk in enumerate(chunks, start=1)
    )
    return hit, recall, reciprocal_rank, retrieved


def _expected_support(case: BenchmarkCase) -> SupportStatus:
    return {
        "answerable": SupportStatus.SUPPORTED,
        "unsupported": SupportStatus.INSUFFICIENT,
        "conflict": SupportStatus.CONFLICTING,
    }[case.type]


def _metrics(results: Sequence[BenchmarkCaseResult]) -> BenchmarkMetrics:
    evidence_cases = [result for result in results if result.type != "unsupported"]
    supported = [result for result in results if result.type == "answerable"]
    unsupported = [result for result in results if result.type == "unsupported"]
    conflicts = [result for result in results if result.type == "conflict"]
    support_run = any(result.support_status is not None for result in results)
    citation_checks = [result for result in results if result.citation_valid is not None]
    generation_attempts = [
        result
        for result in results
        if result.final_status is not None
        and result.support_status == SupportStatus.SUPPORTED.value
    ]
    return BenchmarkMetrics(
        retrieval_hit_at={
            k: sum(result.retrieval_hit_at[k] for result in evidence_cases)
            / len(evidence_cases)
            if evidence_cases
            else 0.0
            for k in _K_VALUES
        },
        retrieval_recall_at={
            k: sum(float(result.retrieval_recall_at[k]) for result in evidence_cases)
            / len(evidence_cases)
            if evidence_cases
            else 0.0
            for k in _K_VALUES
        },
        retrieval_mrr=(
            sum(result.reciprocal_rank or 0.0 for result in evidence_cases)
            / len(evidence_cases)
            if evidence_cases
            else 0.0
        ),
        support_accept_rate=(
            sum(result.support_status == SupportStatus.SUPPORTED.value for result in supported)
            / len(supported)
            if support_run and supported
            else None
        ),
        false_support_rate=(
            sum(result.support_status == SupportStatus.SUPPORTED.value for result in unsupported)
            / len(unsupported)
            if support_run and unsupported
            else None
        ),
        conflict_detection_rate=(
            sum(result.support_status == SupportStatus.CONFLICTING.value for result in conflicts)
            / len(conflicts)
            if support_run and conflicts
            else None
        ),
        citation_validity_rate=(
            sum(result.citation_valid for result in citation_checks) / len(citation_checks)
            if citation_checks
            else None
        ),
        unsupported_claim_rate=(
            sum(result.citation_valid is False for result in generation_attempts)
            / len(generation_attempts)
            if generation_attempts
            else None
        ),
    )


def run_benchmark(
    dataset: BenchmarkDataset,
    *,
    course_id: str,
    retriever: Retriever,
    limit: int = 5,
    support: SupportVerificationService | None = None,
    generator: Generator | None = None,
) -> BenchmarkReport:
    if generator is not None and support is None:
        raise ValueError("generation requires support verification")
    if not 5 <= limit <= 100:
        raise ValueError("benchmark retrieval limit must be between 5 and 100")
    answer_service = AnswerGenerationService(support, generator) if generator else None
    results: list[BenchmarkCaseResult] = []
    for case in dataset.cases:
        chunks = retriever.retrieve(course_id=course_id, query=case.question, limit=limit)
        hit, recall, reciprocal_rank, retrieved = _retrieval_result(case, chunks)
        failures: list[BenchmarkFailure] = []
        if case.expected_sources and recall[5] != 1.0:
            failures.append(BenchmarkFailure("retrieval_failure", "expected source was not retrieved"))
        decision = None
        response: FinalResponse | None = None
        if support is not None:
            decision = support.verify_retrieved_evidence(
                course_id=course_id, question=case.question, evidence=chunks
            )
            if decision.status is not _expected_support(case):
                failures.append(
                    BenchmarkFailure(
                        "support_verification_failure",
                        f"expected {_expected_support(case).value}, got {decision.status.value}",
                    )
                )
            if answer_service is not None:
                response = answer_service.answer_verified_evidence(
                    course_id=course_id,
                    question=case.question,
                    evidence=chunks,
                    decision=decision,
                )
                if response.reason == "citation_validation_failed":
                    failures.append(BenchmarkFailure("citation_failure", response.reason))
                elif (
                    decision.status is SupportStatus.SUPPORTED
                    and response.status is FinalStatus.ABSTAINED
                ):
                    failures.append(BenchmarkFailure("generation_failure", response.reason or "abstained"))
        citation_valid = (
            True
            if response is not None and response.status is FinalStatus.ANSWERED
            else False
            if response is not None and response.reason == "citation_validation_failed"
            else None
        )
        results.append(
            BenchmarkCaseResult(
                case.id,
                case.course_id,
                case.question,
                case.type,
                case.category,
                hit,
                recall,
                reciprocal_rank,
                retrieved,
                decision.status.value if decision else None,
                response.status.value if response else None,
                citation_valid,
                tuple(failures),
            )
        )
    provider = retriever.embedding_provider
    return BenchmarkReport(
        dataset.version,
        course_id,
        {
            "provider": provider.provider_name,
            "model": provider.model_name,
            "dimension": provider.dimension,
        },
        limit,
        (
            {"provider": support.verifier.provider_name, "model": support.verifier.model_name}
            if support
            else None
        ),
        (
            {"provider": generator.provider_name, "model": generator.model_name}
            if generator
            else None
        ),
        _metrics(results),
        tuple(results),
    )
