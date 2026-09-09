from pathlib import Path

import pytest

from course_rag_api.cli import main
from course_rag_api.storage import SQLiteStore


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
