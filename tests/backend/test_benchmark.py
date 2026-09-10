import json
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from course_rag_api.benchmark import load_benchmark_dataset, run_benchmark
from course_rag_api.embeddings import DeterministicEmbeddingProvider
from course_rag_api.generation import GeneratedAnswer, GeneratedClaim
from course_rag_api.indexing import IndexingService
from course_rag_api.ingestion import ingest_document
from course_rag_api.retrieval import Retriever
from course_rag_api.storage import SQLiteStore
from course_rag_api.support import SupportDecision, SupportStatus, SupportVerificationService
from course_rag_api.vector_store import QdrantVectorIndex


def _dataset(course_id: str, cases: list[dict[str, object]]) -> dict[str, object]:
    return {"version": "benchmark-v1", "cases": cases}


def _case(
    case_id: str,
    course_id: str,
    question: str,
    case_type: str,
    sources: list[dict[str, object]],
    category: str = "direct_factual",
) -> dict[str, object]:
    return {
        "id": case_id,
        "course_id": course_id,
        "question": question,
        "type": case_type,
        "category": category,
        "expected": {"supporting_sources": sources},
    }


def _source(document: str, start: int = 1) -> dict[str, object]:
    return {
        "document": document,
        "source_type": "section",
        "start": start,
        "end": start,
    }


def _load(tmp_path: Path, payload: dict[str, object]):
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_benchmark_dataset(path)


def _indexed_retriever(tmp_path: Path) -> tuple[str, Retriever]:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    course = store.create_course("Synthetic benchmark")
    first = tmp_path / "gauss.md"
    first.write_text("Gauss law relates electric flux to enclosed charge.")
    second = tmp_path / "conductor.md"
    second.write_text("The electric field inside a conductor is zero at equilibrium.")
    for path in (first, second):
        ingest_document(
            store=store,
            course_id=course.id,
            file_path=path,
            chunk_target_size=50,
            chunk_overlap=5,
            max_document_bytes=10_000,
        )
    provider = DeterministicEmbeddingProvider()
    client = QdrantClient(":memory:")
    index = QdrantVectorIndex(client, "benchmark", provider)
    IndexingService(store, index, provider).index_course(course.id)
    return course.id, Retriever(store, index, provider)


def test_benchmark_schema_rejects_unsupported_sources_and_invalid_types(tmp_path: Path) -> None:
    course_id = "physics2"
    valid = _load(
        tmp_path,
        _dataset(
            course_id,
            [
                _case(
                    "gauss-001",
                    course_id,
                    "State Gauss law.",
                    "answerable",
                    [_source("gauss.md")],
                ),
                _case(
                    "gauss-unsupported",
                    course_id,
                    "Who discovered Gauss law?",
                    "unsupported",
                    [],
                    "near_miss",
                ),
            ],
        ),
    )

    assert valid.version == "benchmark-v1"
    assert valid.cases[0].expected_sources[0].document == "gauss.md"

    with pytest.raises(ValueError, match="unsupported"):
        _load(
            tmp_path,
            _dataset(
                course_id,
                [
                    _case(
                        "bad",
                        course_id,
                        "Who discovered Gauss law?",
                        "unsupported",
                        [_source("gauss.md")],
                    )
                ],
            ),
        )
    with pytest.raises(ValueError, match="invalid benchmark case type"):
        _load(
            tmp_path,
            _dataset(
                course_id,
                [_case("bad", course_id, "Question?", "unknown", [])],
            ),
        )
    with pytest.raises(ValueError, match="one course_id"):
        _load(
            tmp_path,
            _dataset(
                course_id,
                [
                    _case("first", course_id, "State Gauss law.", "answerable", [_source("gauss.md")]),
                    _case("second", "other-course", "State Gauss law.", "answerable", [_source("gauss.md")]),
                ],
            ),
        )


def test_checked_in_synthetic_benchmark_loads() -> None:
    dataset = load_benchmark_dataset(
        Path(__file__).parents[2] / "benchmarks/examples/synthetic-demo.json"
    )

    assert dataset.version == "synthetic-demo-v1"
    assert {case.category for case in dataset.cases} == {
        "direct_factual",
        "concept_explanation",
        "multi_document",
        "near_miss",
        "citation_sensitive",
    }


def test_checked_in_synthetic_benchmark_runs_without_external_services(
    tmp_path: Path,
) -> None:
    course_id, retriever = _indexed_retriever(tmp_path)
    dataset = load_benchmark_dataset(
        Path(__file__).parents[2] / "benchmarks/examples/synthetic-demo.json"
    )

    report = run_benchmark(dataset, course_id=course_id, retriever=retriever)

    assert report.metrics.retrieval_hit_at[1] == 1.0
    assert report.metrics.retrieval_recall_at[5] == 1.0
    assert report.metrics.retrieval_mrr == 1.0


def test_benchmark_runner_reports_retrieval_metrics_and_failures(tmp_path: Path) -> None:
    course_id, retriever = _indexed_retriever(tmp_path)
    dataset = _load(
        tmp_path,
        _dataset(
            "physics2",
            [
                _case(
                    "gauss-001",
                    "physics2",
                    "What does Gauss law relate?",
                    "answerable",
                    [_source("gauss.md")],
                ),
                _case(
                    "field-unsupported",
                    "physics2",
                    "Who discovered the conductor field?",
                    "unsupported",
                    [],
                    "near_miss",
                ),
                _case(
                    "missing-source",
                    "physics2",
                    "What does Gauss law relate?",
                    "answerable",
                    [_source("missing.md")],
                ),
            ],
        ),
    )

    report = run_benchmark(dataset, course_id=course_id, retriever=retriever, limit=5)

    assert report.metrics.retrieval_hit_at[1] == 0.5
    assert report.metrics.retrieval_recall_at[3] == 0.5
    assert report.metrics.retrieval_mrr == 0.5
    assert report.metrics.support_accept_rate is None
    assert report.cases[2].failures[0].category == "retrieval_failure"
    assert '"retrieval_mrr": 0.5' in report.to_json()


def test_benchmark_runner_evaluates_support_and_citation_failures(tmp_path: Path) -> None:
    class ScriptedVerifier:
        provider_name = "scripted"
        model_name = "benchmark-test"

        def __init__(self) -> None:
            self.calls = 0

        def verify(self, *, question: str, evidence, **_: object) -> SupportDecision:
            self.calls += 1
            if "conflict" in question:
                return SupportDecision(
                    SupportStatus.CONFLICTING,
                    "incompatible_facts",
                    tuple(chunk.chunk_id for chunk in evidence[:2]),
                    self.provider_name,
                    self.model_name,
                )
            if "Who" in question:
                return SupportDecision(
                    SupportStatus.INSUFFICIENT,
                    "missing_attribute",
                    (),
                    self.provider_name,
                    self.model_name,
                )
            return SupportDecision(
                SupportStatus.SUPPORTED,
                "explicit_support",
                (evidence[0].chunk_id,),
                self.provider_name,
                self.model_name,
            )

    class InventedCitationGenerator:
        provider_name = "scripted"
        model_name = "benchmark-test"

        def generate(self, **_: object) -> GeneratedAnswer:
            claim = GeneratedClaim("An unsupported answer.", ("invented",))
            return GeneratedAnswer(claim.text, (claim,), self.provider_name, self.model_name)

    course_id, retriever = _indexed_retriever(tmp_path)
    dataset = _load(
        tmp_path,
        _dataset(
            "physics2",
            [
                _case(
                    "gauss-001",
                    "physics2",
                    "What does Gauss law relate?",
                    "answerable",
                    [_source("gauss.md")],
                ),
                _case(
                    "field-unsupported",
                    "physics2",
                    "Who discovered the conductor field?",
                    "unsupported",
                    [],
                    "near_miss",
                ),
                _case(
                    "field-conflict",
                    "physics2",
                    "What is the conflict?",
                    "conflict",
                    [_source("gauss.md"), _source("conductor.md")],
                    "citation_sensitive",
                ),
            ],
        ),
    )
    verifier = ScriptedVerifier()
    support = SupportVerificationService(retriever.store, retriever, verifier)

    report = run_benchmark(
        dataset,
        course_id=course_id,
        retriever=retriever,
        limit=5,
        support=support,
        generator=InventedCitationGenerator(),
    )

    assert report.metrics.support_accept_rate == 1.0
    assert report.metrics.false_support_rate == 0.0
    assert report.metrics.conflict_detection_rate == 1.0
    assert report.metrics.citation_validity_rate == 0.0
    assert report.metrics.unsupported_claim_rate == 1.0
    assert report.cases[0].citation_valid is False
    assert report.cases[0].failures[0].category == "citation_failure"
    assert verifier.calls == len(dataset.cases)


def test_benchmark_local_directory_excludes_real_course_documents() -> None:
    root = Path(__file__).parents[2]
    local_ignore = root / "benchmarks/local/.gitignore"

    assert local_ignore.read_text(encoding="utf-8").strip() == "*\n!.gitignore"
    assert "/benchmarks/local/*" in (root / ".gitignore").read_text(encoding="utf-8")
    assert not [
        path
        for path in (root / "benchmarks").rglob("*")
        if path.suffix.casefold() in {".pdf", ".pptx", ".docx"}
    ]
