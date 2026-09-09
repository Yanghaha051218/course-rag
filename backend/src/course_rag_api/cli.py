import argparse
import sys
from pathlib import Path
from typing import Sequence

from course_rag_api.config import get_settings
from course_rag_api.errors import CourseRAGError
from course_rag_api.ingestion import ingest_document
from course_rag_api.storage import SQLiteStore


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
    return parser


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
    except (CourseRAGError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
