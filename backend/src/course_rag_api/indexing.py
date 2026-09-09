from course_rag_api.embeddings import EmbeddingProvider
from course_rag_api.models import Chunk, IndexingSummary
from course_rag_api.storage import SQLiteStore
from course_rag_api.vector_store import QdrantVectorIndex


class IndexingService:
    """Embed only SQLite chunks not already present in the vector collection."""

    def __init__(
        self,
        store: SQLiteStore,
        vector_index: QdrantVectorIndex,
        embedding_provider: EmbeddingProvider,
        *,
        batch_size: int = 64,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("embedding batch size must be positive")
        vector_index.require_compatible_provider(embedding_provider)
        self.store = store
        self.vector_index = vector_index
        self.embedding_provider = embedding_provider
        self.batch_size = batch_size

    def index_document(self, document_id: str) -> IndexingSummary:
        document = self.store.get_document(document_id)
        chunks = self.store.list_document_chunks(document.course_id, document.id)
        return self._index(
            course_id=document.course_id,
            document_id=document.id,
            chunks=chunks,
            filenames={document.id: document.filename},
        )

    def index_course(self, course_id: str) -> IndexingSummary:
        self.store.get_course(course_id)
        # ponytail: load one course at once; paginate if course-size profiling needs it.
        chunks = self.store.list_chunks(course_id)
        filenames = {
            document.id: document.filename
            for document in self.store.list_documents(course_id)
        }
        return self._index(
            course_id=course_id,
            document_id=None,
            chunks=chunks,
            filenames=filenames,
        )

    def _index(
        self,
        *,
        course_id: str,
        document_id: str | None,
        chunks: tuple[Chunk, ...],
        filenames: dict[str, str],
    ) -> IndexingSummary:
        missing = self.vector_index.missing_chunk_ids([chunk.id for chunk in chunks])
        pending = [chunk for chunk in chunks if chunk.id in missing]
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            vectors = self.embedding_provider.embed_documents(
                [chunk.text for chunk in batch]
            )
            self.vector_index.upsert(batch, vectors, filenames)
        return IndexingSummary(
            course_id=course_id,
            document_id=document_id,
            total_count=len(chunks),
            indexed_count=len(pending),
            collection_name=self.vector_index.collection_name,
        )
