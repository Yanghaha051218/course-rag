import re
from dataclasses import dataclass
from hashlib import sha256
from qdrant_client import QdrantClient, models

from course_rag_api.embeddings import EmbeddingProvider
from course_rag_api.errors import IndexCompatibilityError, IndexConsistencyError
from course_rag_api.models import Chunk, SourceType


@dataclass(frozen=True, slots=True)
class VectorMatch:
    chunk_id: str
    course_id: str
    document_id: str
    chunk_index: int
    source_type: SourceType
    source_start: int
    source_end: int
    filename: str
    score: float


class QdrantVectorIndex:
    """Qdrant operations for one embedding provider/model/dimension."""

    def __init__(
        self,
        client: QdrantClient,
        collection_prefix: str,
        provider: EmbeddingProvider,
    ) -> None:
        prefix = re.sub(r"[^a-zA-Z0-9_-]+", "_", collection_prefix).strip("_")
        if not prefix:
            raise ValueError("collection prefix must contain letters or numbers")
        self.client = client
        self.provider_name = provider.provider_name
        self.model_name = provider.model_name
        self.dimension = provider.dimension
        identity = f"{self.provider_name}:{self.model_name}:{self.dimension}"
        self.collection_name = f"{prefix}_{sha256(identity.encode()).hexdigest()[:12]}"

    @property
    def _metadata(self) -> dict[str, str | int]:
        return {
            "embedding_provider": self.provider_name,
            "embedding_model": self.model_name,
            "embedding_dimension": self.dimension,
        }

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimension, distance=models.Distance.COSINE
                ),
                metadata=self._metadata,
            )
            return
        collection = self.client.get_collection(self.collection_name)
        vectors = collection.config.params.vectors
        if (
            not isinstance(vectors, models.VectorParams)
            or vectors.size != self.dimension
            or collection.config.metadata != self._metadata
        ):
            raise IndexCompatibilityError(
                f"Collection {self.collection_name} has incompatible embedding metadata"
            )

    def require_compatible_provider(self, provider: EmbeddingProvider) -> None:
        if (
            provider.provider_name,
            provider.model_name,
            provider.dimension,
        ) != (self.provider_name, self.model_name, self.dimension):
            raise IndexCompatibilityError(
                "embedding provider does not match the vector collection"
            )

    def missing_chunk_ids(self, chunk_ids: list[str]) -> set[str]:
        if not chunk_ids:
            return set()
        self.ensure_collection()
        existing = self.client.retrieve(
            collection_name=self.collection_name,
            ids=chunk_ids,
            with_payload=False,
            with_vectors=False,
        )
        return set(chunk_ids) - {str(point.id) for point in existing}

    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        filenames: dict[str, str],
    ) -> None:
        if len(chunks) != len(vectors):
            raise IndexConsistencyError("chunk and embedding counts do not match")
        if any(len(vector) != self.dimension for vector in vectors):
            raise IndexCompatibilityError("embedding dimension does not match collection")
        self.ensure_collection()
        self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=[
                models.PointStruct(
                    id=chunk.id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk.id,
                        "course_id": chunk.course_id,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "source_type": chunk.source_type,
                        "source_start": chunk.source_start,
                        "source_end": chunk.source_end,
                        "filename": filenames[chunk.document_id],
                    },
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )

    def delete_chunk_ids(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        self.ensure_collection()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=chunk_ids),
            wait=True,
        )

    def search(
        self, *, course_id: str, vector: list[float], limit: int
    ) -> tuple[VectorMatch, ...]:
        if not course_id.strip():
            raise ValueError("course_id is required")
        if not 1 <= limit <= 100:
            raise ValueError("retrieval limit must be between 1 and 100")
        if len(vector) != self.dimension:
            raise IndexCompatibilityError("query embedding dimension is incompatible")
        self.ensure_collection()
        points = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="course_id", match=models.MatchValue(value=course_id)
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        ).points
        matches = [self._match(point, course_id) for point in points]
        return tuple(sorted(matches, key=lambda item: (-item.score, item.chunk_id)))

    @staticmethod
    def _match(
        point: models.ScoredPoint, requested_course_id: str
    ) -> VectorMatch:
        payload = point.payload or {}
        required = {
            "chunk_id",
            "course_id",
            "document_id",
            "chunk_index",
            "source_type",
            "source_start",
            "source_end",
            "filename",
        }
        if not required.issubset(payload):
            raise IndexConsistencyError(f"Qdrant point {point.id} has incomplete payload")
        if payload["course_id"] != requested_course_id:
            raise IndexConsistencyError("Qdrant returned a cross-course point")
        if payload["chunk_id"] != str(point.id):
            raise IndexConsistencyError("Qdrant point ID and chunk_id do not match")
        try:
            return VectorMatch(score=float(point.score), **payload)
        except (TypeError, ValueError) as error:
            raise IndexConsistencyError(
                f"Qdrant point {point.id} has invalid payload"
            ) from error
