from pathlib import Path

import pytest

from course_rag_api.calibration import (
    CalibrationIdentity,
    LabeledScore,
    calibrate_threshold,
)
from course_rag_api.cli import main
from course_rag_api.config import get_settings
from course_rag_api.storage import SQLiteStore
from course_rag_api.support import SupportDecision, SupportStatus


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

    assert "Cases: 40 (answerable=20, unsupported=20)" in output
    assert "Hit@1:" in output
    assert "Hit@3:" in output
    assert "Hit@5:" in output
    assert "Isolation: PASS" in output
    assert '"case_count": 40' in output_path.read_text()
    get_settings.cache_clear()


def test_cli_calibrates_retrieval_with_separate_holdout_metrics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "calibration.json"
    monkeypatch.setenv("COURSE_RAG_EMBEDDING_PROVIDER", "deterministic")
    get_settings.cache_clear()

    assert (
        main(
            [
                "calibrate-retrieval",
                str(EVALUATION_DATASET),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "Selected policy: top1_threshold" in output
    assert "CALIBRATION" in output
    assert "HOLDOUT" in output
    assert "Unsupported false-accept:" in output
    assert '"dataset_version": "m2b-v1"' in output_path.read_text()
    get_settings.cache_clear()


def test_cli_inspects_evidence_without_generating_an_answer(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "metadata.sqlite3"
    calibration_path = tmp_path / "calibration.json"
    monkeypatch.setenv("COURSE_RAG_QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("COURSE_RAG_EMBEDDING_PROVIDER", "deterministic")
    get_settings.cache_clear()
    calibration_path.write_text(
        calibrate_threshold(
            (
                LabeledScore("answer", True, 0.2, 0.2),
                LabeledScore("unsupported", False, 0.1, 0.1),
            ),
            CalibrationIdentity(
                "deterministic",
                "deterministic-hash-v1",
                256,
                "test-v1",
                5,
                32,
                4,
            ),
        ).to_json()
    )
    main(["--database", str(database), "create-course", "Evidence Demo"])
    course_id = capsys.readouterr().out.split("Course ID: ", 1)[1].strip()
    path = tmp_path / "lesson.txt"
    path.write_text("The fictional valve opens after twelve measured minutes.")
    main(["--database", str(database), "ingest", "--course", course_id, str(path)])
    capsys.readouterr()
    main(["--database", str(database), "index-course", "--course", course_id])
    capsys.readouterr()

    assert (
        main(
            [
                "--database",
                str(database),
                "inspect-evidence",
                "--course",
                course_id,
                "--query",
                "When does the fictional valve open?",
                "--calibration",
                str(calibration_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "Decision: ALLOW" in output
    assert "Evidence:" in output
    assert "lesson.txt" in output
    assert "answer" not in output.casefold()
    get_settings.cache_clear()


def test_cli_verifies_support_without_printing_an_answer(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ScriptedSupportVerifier:
        provider_name = "scripted"
        model_name = "test-v1"

        def verify(self, **_: object) -> SupportDecision:
            return SupportDecision(
                SupportStatus.SUPPORTED,
                "explicit_support",
                (chunk_id,),
                self.provider_name,
                self.model_name,
            )

    database = tmp_path / "metadata.sqlite3"
    monkeypatch.setenv("COURSE_RAG_QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("COURSE_RAG_EMBEDDING_PROVIDER", "deterministic")
    get_settings.cache_clear()
    main(["--database", str(database), "create-course", "Support Demo"])
    course_id = capsys.readouterr().out.split("Course ID: ", 1)[1].strip()
    path = tmp_path / "lesson.txt"
    path.write_text("The Blueleaf coefficient is 7.25.")
    main(["--database", str(database), "ingest", "--course", course_id, str(path)])
    capsys.readouterr()
    main(["--database", str(database), "index-course", "--course", course_id])
    capsys.readouterr()
    chunk_id = SQLiteStore(database).list_chunks(course_id)[0].id
    monkeypatch.setattr(
        "course_rag_api.cli.create_support_verifier",
        lambda _: ScriptedSupportVerifier(),
    )

    assert (
        main(
            [
                "--database",
                str(database),
                "verify-support",
                "--course",
                course_id,
                "--query",
                "What is the Blueleaf coefficient?",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "Decision: SUPPORTED" in output
    assert "Supporting evidence:" in output
    assert chunk_id in output
    assert "7.25" not in output
    get_settings.cache_clear()
