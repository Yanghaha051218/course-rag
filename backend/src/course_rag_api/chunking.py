import re
from collections.abc import Sequence
from uuid import NAMESPACE_URL, uuid5

from course_rag_api.models import Chunk, SourceType, SourceUnit


_SENTENCE_BREAK = re.compile(r"(?<=[.!?。！？])\s+|\n+")
_WordRef = tuple[str, SourceType, int]


def _word_refs(units: Sequence[SourceUnit]) -> tuple[list[_WordRef], set[int]]:
    words: list[_WordRef] = []
    boundaries: set[int] = set()
    for unit in units:
        for sentence in _SENTENCE_BREAK.split(unit.text):
            words.extend(
                (word, unit.source_type, unit.source_number)
                for word in sentence.split()
            )
            if words:
                boundaries.add(len(words))
    return words, boundaries


def chunk_source_units(
    *,
    course_id: str,
    document_id: str,
    units: Sequence[SourceUnit],
    target_size: int,
    overlap: int,
    created_at: str,
) -> tuple[Chunk, ...]:
    """Create deterministic word-budget chunks, preferring sentence endings."""
    if target_size <= 0 or overlap < 0 or overlap >= target_size:
        raise ValueError("chunk target size must be positive and exceed overlap")
    if len({unit.source_type for unit in units}) > 1:
        raise ValueError("source units in one document must share a source type")

    words, boundaries = _word_refs(units)
    chunks: list[Chunk] = []
    start = 0
    previous_end = 0
    while start < len(words):
        desired_end = min(start + target_size, len(words))
        natural_ends = [
            boundary
            for boundary in boundaries
            if previous_end < boundary <= desired_end
        ]
        end = max(natural_ends, default=desired_end)
        window = words[start:end]
        text = " ".join(word for word, _, _ in window)
        source_type = window[0][1]
        chunk_index = len(chunks)
        chunks.append(
            Chunk(
                id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"course-rag:{document_id}:{chunk_index}:{text}",
                    )
                ),
                course_id=course_id,
                document_id=document_id,
                text=text,
                chunk_index=chunk_index,
                source_type=source_type,
                source_start=window[0][2],
                source_end=window[-1][2],
                created_at=created_at,
            )
        )
        if end == len(words):
            break
        previous_end = end
        start = max(end - overlap, start + 1)
    return tuple(chunks)
