import argparse
import sys
from pathlib import Path
from typing import Sequence

from qdrant_client import QdrantClient

from course_rag_api.config import get_settings
from course_rag_api.embeddings import create_embedding_provider
from course_rag_api.errors import CourseRAGError
from course_rag_api.indexing import IndexingService
from course_rag_api.ingestion import ingest_document
from course_rag_api.models import IndexingSummary
from course_rag_api.retrieval import Retriever
from course_rag_api.storage import SQLiteStore
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
    return parser


def _print_index_summary(summary: IndexingSummary) -> None:
    print(f"Course: {summary.course_id}")
    if summary.document_id is not None:
        print(f"Document: {summary.document_id}")
    print(f"Collection: {summary.collection_name}")
    print(f"Chunks total: {summary.total_count}")
    print(f"Indexed now: {summary.indexed_count}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()
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
