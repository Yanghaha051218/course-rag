import argparse
import sys
from pathlib import Path
from typing import Sequence

from qdrant_client import QdrantClient

from course_rag_api.calibration import (
    CalibrationIdentity,
    CalibrationResult,
    LabeledScore,
    calibrate_margin_threshold,
    calibrate_threshold,
    evaluate_policy,
    summarize_distribution,
)
from course_rag_api.config import get_settings
from course_rag_api.embeddings import create_embedding_provider
from course_rag_api.evidence import EvidenceGate
from course_rag_api.errors import CourseRAGError
from course_rag_api.evaluation import (
    format_evaluation_report,
    load_evaluation_dataset,
    run_retrieval_evaluation,
)
from course_rag_api.indexing import IndexingService
from course_rag_api.ingestion import ingest_document
from course_rag_api.models import IndexingSummary
from course_rag_api.retrieval import Retriever
from course_rag_api.storage import SQLiteStore
from course_rag_api.support import SupportVerificationService, create_support_verifier
from course_rag_api.vector_store import QdrantVectorIndex


def _parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="course-rag")
    parser.add_argument("--database", type=Path, default=settings.database_path)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create-course")
    create.add_argument("name")
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--course", required=True)
    ingest.add_argument("file", type=Path)
    index_document = commands.add_parser("index-document")
    index_document.add_argument("--document", required=True)
    index_course = commands.add_parser("index-course")
    index_course.add_argument("--course", required=True)
    retrieve = commands.add_parser("retrieve")
    retrieve.add_argument("--course", required=True)
    retrieve.add_argument("--query", required=True)
    retrieve.add_argument("--limit", type=int, default=5)
    evaluate = commands.add_parser("evaluate-retrieval")
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("--limit", type=int, default=5)
    evaluate.add_argument("--output", type=Path)
    calibrate = commands.add_parser("calibrate-retrieval")
    calibrate.add_argument("dataset", type=Path)
    calibrate.add_argument("--limit", type=int, default=5)
    calibrate.add_argument("--output", type=Path)
    inspect = commands.add_parser("inspect-evidence")
    inspect.add_argument("--course", required=True)
    inspect.add_argument("--query", required=True)
    inspect.add_argument("--calibration", required=True, type=Path)
    inspect.add_argument("--limit", type=int, default=5)
    verify_support = commands.add_parser("verify-support")
    verify_support.add_argument("--course", required=True)
    verify_support.add_argument("--query", required=True)
    verify_support.add_argument("--limit", type=int, default=5)
    return parser


def _print_index_summary(summary: IndexingSummary) -> None:
    print(f"Course: {summary.course_id}")
    if summary.document_id is not None:
        print(f"Document: {summary.document_id}")
    print(f"Collection: {summary.collection_name}")
    print(f"Chunks total: {summary.total_count}")
    print(f"Indexed now: {summary.indexed_count}")


def _labeled_scores(report, split: str) -> tuple[LabeledScore, ...]:
    return tuple(
        LabeledScore(
            case.label,
            case.answerable,
            case.top_1_score,
            case.top_1_top_2_margin,
        )
        for case in report.cases
        if case.split == split
    )


def _print_metrics(label: str, metrics) -> None:
    print(label)
    print(f"  Answerable allow: {metrics.answerable_allow}")
    print(f"  Answerable abstain: {metrics.answerable_abstain}")
    print(f"  Unsupported allow: {metrics.unsupported_allow}")
    print(f"  Unsupported abstain: {metrics.unsupported_abstain}")
    print(f"  Answerable acceptance: {metrics.answerable_acceptance_rate:.3f}")
    print(f"  Unsupported false-accept: {metrics.unsupported_false_accept_rate:.3f}")
    print(f"  Abstention: {metrics.abstention_rate:.3f}")


def _print_distribution(label: str, values: list[float]) -> None:
    summary = summarize_distribution(values)
    print(
        f"{label}: count={summary.count} min={summary.minimum!s} "
        f"max={summary.maximum!s} mean={summary.mean!s} median={summary.median!s} "
        f"p10={summary.p10!s} p25={summary.p25!s} "
        f"p75={summary.p75!s} p90={summary.p90!s}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()
    if args.command in {"evaluate-retrieval", "calibrate-retrieval"}:
        try:
            dataset = load_evaluation_dataset(args.dataset)
            provider = create_embedding_provider(settings)
            report = run_retrieval_evaluation(dataset, provider, limit=args.limit)
            if args.command == "evaluate-retrieval":
                print(format_evaluation_report(report))
                if args.output is not None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(report.to_json(), encoding="utf-8")
                    print(f"JSON: {args.output}")
                return 0

            identity = CalibrationIdentity(
                provider.provider_name,
                provider.model_name,
                provider.dimension,
                dataset.version,
                args.limit,
                dataset.chunk_target_size,
                dataset.chunk_overlap,
            )
            calibration_cases = _labeled_scores(report, "calibration")
            holdout_cases = _labeled_scores(report, "holdout")
            calibration = calibrate_threshold(calibration_cases, identity)
            holdout_metrics = evaluate_policy(
                holdout_cases, threshold=calibration.threshold
            )
            margin = calibrate_margin_threshold(
                calibration_cases, threshold=calibration.threshold
            )
            print(f"Provider: {provider.provider_name}")
            print(f"Model: {provider.model_name}")
            print(f"Dimension: {provider.dimension}")
            print(f"Dataset version: {dataset.version}")
            print(f"Selected policy: {calibration.policy_type}")
            print(f"Selected threshold: {calibration.threshold:.6f}")
            _print_metrics("CALIBRATION", calibration.calibration_metrics)
            _print_metrics("HOLDOUT", holdout_metrics)
            _print_distribution(
                "Answerable top-1",
                [
                    case.top_1_score
                    for case in report.cases
                    if case.answerable and case.top_1_score is not None
                ],
            )
            _print_distribution(
                "Unsupported top-1",
                [
                    case.top_1_score
                    for case in report.cases
                    if not case.answerable and case.top_1_score is not None
                ],
            )
            _print_distribution(
                "Answerable margin",
                [
                    case.top_1_top_2_margin
                    for case in report.cases
                    if case.answerable and case.top_1_top_2_margin is not None
                ],
            )
            _print_distribution(
                "Unsupported margin",
                [
                    case.top_1_top_2_margin
                    for case in report.cases
                    if not case.answerable and case.top_1_top_2_margin is not None
                ],
            )
            if margin.metrics is None:
                print("Margin comparison: unavailable after top-1 threshold")
            else:
                holdout_margin = evaluate_policy(
                    holdout_cases,
                    threshold=calibration.threshold,
                    margin_threshold=margin.margin_threshold,
                )
                print(
                    "Margin comparison: baseline remains selected; "
                    f"candidate margin={margin.margin_threshold:.6f}"
                )
                _print_metrics("HOLDOUT margin candidate", holdout_margin)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(calibration.to_json(), encoding="utf-8")
                print(f"Calibration JSON: {args.output}")
            return 0
        except (CourseRAGError, OSError, ValueError) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
    store = SQLiteStore(args.database)
    try:
        if args.command == "create-course":
            course = store.create_course(args.name)
            print(f"Course: {course.name}")
            print(f"Course ID: {course.id}")
            return 0

        if args.command == "ingest":
            summary = ingest_document(
                store=store,
                course_id=args.course,
                file_path=args.file,
                chunk_target_size=settings.chunk_target_size,
                chunk_overlap=settings.chunk_overlap,
                max_document_bytes=settings.max_document_bytes,
            )
            print(f"Document: {summary.filename}")
            print(f"Type: {summary.file_type}")
            print(f"Course: {summary.course_id}")
            print(f"Source units: {summary.source_unit_count}")
            print(f"Chunks: {summary.chunk_count}")
            print(f"Status: {summary.status}")
            return 0

        provider = create_embedding_provider(settings)
        client = QdrantClient(path=str(settings.qdrant_path))
        try:
            index = QdrantVectorIndex(
                client, settings.qdrant_collection_prefix, provider
            )
            if args.command == "index-document":
                _print_index_summary(
                    IndexingService(
                        store,
                        index,
                        provider,
                        batch_size=settings.embedding_batch_size,
                    ).index_document(args.document)
                )
                return 0
            if args.command == "index-course":
                _print_index_summary(
                    IndexingService(
                        store,
                        index,
                        provider,
                        batch_size=settings.embedding_batch_size,
                    ).index_course(args.course)
                )
                return 0
            if args.command == "inspect-evidence":
                calibration = CalibrationResult.from_json(args.calibration)
                if calibration.identity.retrieval_limit != args.limit:
                    raise ValueError(
                        "calibration retrieval_limit does not match "
                        "inspect-evidence limit"
                    )
                results = Retriever(store, index, provider).retrieve(
                    course_id=args.course, query=args.query, limit=args.limit
                )
                decision = EvidenceGate(provider).evaluate(
                    course_id=args.course,
                    retrieved_chunks=results,
                    calibration=calibration,
                )
                print(f"Decision: {decision.status.upper()}")
                print(f"Reason: {decision.reason}")
                print(f"Top score: {decision.top_score}")
                print(f"Threshold: {decision.threshold:.6f}")
                if decision.evidence:
                    print("Evidence:")
                    for chunk in decision.evidence:
                        location = (
                            f"{chunk.source_type} {chunk.source_start}"
                            if chunk.source_start == chunk.source_end
                            else f"{chunk.source_type} "
                            f"{chunk.source_start}-{chunk.source_end}"
                        )
                        print(f"- {chunk.filename} {location} score={chunk.score:.6f}")
                return 0
            if args.command == "verify-support":
                retriever = Retriever(store, index, provider)
                results = retriever.retrieve(
                    course_id=args.course, query=args.query, limit=args.limit
                )
                verifier = create_support_verifier(settings)
                decision = SupportVerificationService(
                    store, retriever, verifier, args.limit
                ).verify_retrieved_evidence(
                    course_id=args.course,
                    question=args.query,
                    evidence=results,
                )
                print(f"Decision: {decision.status.value}")
                print(f"Reason: {decision.reason}")
                print(
                    f"Verifier: {decision.verifier_provider} / "
                    f"{decision.verifier_model}"
                )
                if results:
                    print("Retrieved evidence:")
                    for result in results:
                        print(
                            f"- {result.filename} {result.source_type} "
                            f"{result.source_start}-{result.source_end} "
                            f"score={result.score:.6f} chunk_id={result.chunk_id}"
                        )
                if decision.supporting_chunk_ids:
                    print("Supporting evidence:")
                    for chunk_id in decision.supporting_chunk_ids:
                        print(f"- chunk_id={chunk_id}")
                return 0
            results = Retriever(store, index, provider).retrieve(
                course_id=args.course, query=args.query, limit=args.limit
            )
            if not results:
                print("No indexed chunks found for this course.")
            for rank, result in enumerate(results, start=1):
                location = (
                    f"{result.source_type} {result.source_start}"
                    if result.source_start == result.source_end
                    else f"{result.source_type} {result.source_start}-{result.source_end}"
                )
                print(f"{rank}. score={result.score:.6f}")
                print(f"   {result.filename}")
                print(f"   {location}")
                print(f"   {result.text}")
            return 0
        finally:
            client.close()
    except (CourseRAGError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
