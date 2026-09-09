from pathlib import Path

import pytest

from course_rag_api.cli import main
from course_rag_api.config import get_settings
from course_rag_api.storage import SQLiteStore


EVALUATION_DATASET = (
    Path(__file__).parents[2] / "examples/retrieval-evaluation/dataset.json"
)


def test_cli_creates_course_and_ingests_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "metadata.sqlite3"
    assert main(["--database", str(database), "create-course", "Physics Demo"]) == 0
    output = capsys.readouterr().out
    course_id = output.split("Course ID: ", 1)[1].strip()
    path = tmp_path / "lesson.txt"
    path.write_text("A synthetic orbit lasts exactly eleven minutes.")

    assert (
        main(
            [
                "--database",
                str(database),
                "ingest",
                "--course",
                course_id,
                str(path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "Document: lesson.txt" in output
    assert "Status: ingested" in output
    assert len(SQLiteStore(database).list_chunks(course_id)) == 1


def test_cli_indexes_and_retrieves_without_external_services(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "metadata.sqlite3"
    monkeypatch.setenv("COURSE_RAG_QDRANT_PATH", str(tmp_path / "qdrant"))
    get_settings.cache_clear()
    main(["--database", str(database), "create-course", "Orbital Gardening"])
    course_id = capsys.readouterr().out.split("Course ID: ", 1)[1].strip()
    path = tmp_path / "blueleaf.txt"
    path.write_text("The Blueleaf coefficient is 7.25.")
    main(
        [
            "--database",
            str(database),
            "ingest",
            "--course",
            course_id,
            str(path),
        ]
    )
    capsys.readouterr()

    assert (
        main(
            [
                "--database",
                str(database),
                "index-course",
                "--course",
                course_id,
            ]
        )
        == 0
    )
    assert "Indexed now: 1" in capsys.readouterr().out
    assert (
        main(
            [
                "--database",
                str(database),
                "retrieve",
                "--course",
                course_id,
                "--query",
                "What is the Blueleaf coefficient?",
                "--limit",
                "1",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "blueleaf.txt" in output
    assert "section 1" in output
    assert "The Blueleaf coefficient is 7.25." in output
    get_settings.cache_clear()


def test_cli_evaluates_retrieval_and_writes_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "evaluation.json"
    monkeypatch.setenv("COURSE_RAG_EMBEDDING_PROVIDER", "deterministic")
    monkeypatch.delenv("COURSE_RAG_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("COURSE_RAG_EMBEDDING_DIMENSION", raising=False)
    get_settings.cache_clear()

    assert (
        main(
            [
                "evaluate-retrieval",
                str(EVALUATION_DATASET),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "Cases: 8 (answerable=6, unsupported=2)" in output
    assert "Hit@1:" in output
    assert "Hit@3:" in output
    assert "Hit@5:" in output
    assert "Isolation: PASS" in output
    assert '"case_count": 8' in output_path.read_text()
    get_settings.cache_clear()
