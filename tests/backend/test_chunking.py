from course_rag_api.chunking import chunk_source_units
from course_rag_api.models import SourceUnit


def test_chunking_is_deterministic_bounded_and_preserves_overlap() -> None:
    units = (
        SourceUnit("Alpha beta gamma delta.", "page", 1),
        SourceUnit("Epsilon zeta eta theta.", "page", 2),
    )
    arguments = {
        "course_id": "course-a",
        "document_id": "document-a",
        "units": units,
        "target_size": 5,
        "overlap": 2,
        "created_at": "2026-01-01T00:00:00+00:00",
    }

    chunks = chunk_source_units(**arguments)

    assert chunks == chunk_source_units(**arguments)
    assert chunks[0].text == "Alpha beta gamma delta."
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.text and len(chunk.text.split()) <= 5 for chunk in chunks)
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]
    assert (chunks[1].source_start, chunks[1].source_end) == (1, 2)
    assert all(chunk.course_id == "course-a" for chunk in chunks)
    assert all(chunk.document_id == "document-a" for chunk in chunks)


def test_chunking_retains_slide_source_ranges() -> None:
    chunks = chunk_source_units(
        course_id="course-a",
        document_id="slides-a",
        units=(
            SourceUnit("One two three four.", "slide", 8),
            SourceUnit("Five six seven eight.", "slide", 9),
        ),
        target_size=6,
        overlap=1,
        created_at="2026-01-01T00:00:00+00:00",
    )

    assert chunks[1].source_type == "slide"
    assert chunks[1].source_start <= chunks[1].source_end


def test_chunking_rejects_invalid_sizes() -> None:
    units = (SourceUnit("Some source text.", "section", 1),)

    for target_size, overlap in ((0, 0), (5, -1), (5, 5)):
        try:
            chunk_source_units(
                course_id="course-a",
                document_id="document-a",
                units=units,
                target_size=target_size,
                overlap=overlap,
                created_at="2026-01-01T00:00:00+00:00",
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid chunk settings must fail")
