from course_rag_api.embeddings import EmbeddingProvider
from course_rag_api.errors import IndexConsistencyError
from course_rag_api.models import RetrievedChunk
from course_rag_api.storage import SQLiteStore
from course_rag_api.vector_store import QdrantVectorIndex


class Retriever:
    def __init__(
        self,
        store: SQLiteStore,
        vector_index: QdrantVectorIndex,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        vector_index.require_compatible_provider(embedding_provider)
        self.store = store
        self.vector_index = vector_index
        self.embedding_provider = embedding_provider

    def retrieve(
        self, *, course_id: str, query: str, limit: int = 5
    ) -> tuple[RetrievedChunk, ...]:
        """Return raw, ranked evidence; no threshold or answer generation."""
        if not 1 <= limit <= 100:
            raise ValueError("retrieval limit must be between 1 and 100")
        self.store.get_course(course_id)
        matches = self.vector_index.search(
            course_id=course_id,
            vector=self.embedding_provider.embed_query(query),
            limit=limit,
        )
        records = self.store.get_chunks_with_filenames(
            course_id, [match.chunk_id for match in matches]
        )
        missing = [match.chunk_id for match in matches if match.chunk_id not in records]
        if missing:
            raise IndexConsistencyError(
                f"Qdrant chunks are missing from SQLite: {', '.join(missing)}"
            )
        results: list[RetrievedChunk] = []
        for match in matches:
            chunk, filename = records[match.chunk_id]
            if (
                match.document_id != chunk.document_id
                or match.chunk_index != chunk.chunk_index
                or match.source_type != chunk.source_type
                or match.source_start != chunk.source_start
                or match.source_end != chunk.source_end
                or match.filename != filename
            ):
                raise IndexConsistencyError(
                    f"Qdrant provenance disagrees with SQLite for chunk {chunk.id}"
                )
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    course_id=chunk.course_id,
                    document_id=chunk.document_id,
                    filename=filename,
                    text=chunk.text,
                    score=match.score,
                    chunk_index=chunk.chunk_index,
                    source_type=chunk.source_type,
                    source_start=chunk.source_start,
                    source_end=chunk.source_end,
                )
            )
        return tuple(results)
