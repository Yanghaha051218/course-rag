from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from course_rag_api.embeddings import DeterministicEmbeddingProvider
from course_rag_api.errors import (
    CourseNotFoundError,
    IndexCompatibilityError,
    IndexConsistencyError,
)
from course_rag_api.indexing import IndexingService
from course_rag_api.ingestion import ingest_document
from course_rag_api.retrieval import Retriever
from course_rag_api.storage import SQLiteStore
from course_rag_api.vector_store import QdrantVectorIndex


def _ingest(store: SQLiteStore, course_id: str, path: Path, text: str) -> str:
    path.write_text(text)
    return ingest_document(
        store=store,
        course_id=course_id,
        file_path=path,
        chunk_target_size=40,
        chunk_overlap=5,
    ).document_id


def _services(
    tmp_path: Path,
) -> tuple[
    SQLiteStore,
    DeterministicEmbeddingProvider,
    QdrantClient,
    QdrantVectorIndex,
    IndexingService,
    Retriever,
]:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    provider = DeterministicEmbeddingProvider(dimension=64)
    client = QdrantClient(location=":memory:")
    index = QdrantVectorIndex(client, "course_rag_test", provider)
    return (
        store,
        provider,
        client,
        index,
        IndexingService(store, index, provider, batch_size=2),
        Retriever(store, index, provider),
    )


def test_indexing_preserves_payload_dimension_and_is_idempotent(
    tmp_path: Path,
) -> None:
    store, provider, client, index, indexing, _ = _services(tmp_path)
    course = store.create_course("Orbital Gardening")
    document_id = _ingest(
        store,
        course.id,
        tmp_path / "blueleaf.txt",
        "The Blueleaf coefficient is 7.25.",
    )
    chunk = store.list_chunks(course.id)[0]

    first = indexing.index_document(document_id)
    second = indexing.index_document(document_id)
    point = client.retrieve(index.collection_name, [chunk.id])[0]
    collection = client.get_collection(index.collection_name)

    assert (first.indexed_count, first.total_count) == (1, 1)
    assert (second.indexed_count, second.total_count) == (0, 1)
    assert client.count(index.collection_name, exact=True).count == 1
    assert str(point.id) == chunk.id
    assert point.payload == {
        "chunk_id": chunk.id,
        "course_id": course.id,
        "document_id": document_id,
        "chunk_index": 0,
        "source_type": "section",
        "source_start": 1,
        "source_end": 1,
        "filename": "blueleaf.txt",
    }
    assert collection.config.params.vectors.size == provider.dimension
    assert collection.config.metadata == {
        "embedding_provider": provider.provider_name,
        "embedding_model": provider.model_name,
        "embedding_dimension": provider.dimension,
    }


def test_empty_course_indexes_nothing(tmp_path: Path) -> None:
    store, _, _, _, indexing, _ = _services(tmp_path)
    course = store.create_course("Empty Course")

    summary = indexing.index_course(course.id)

    assert summary.indexed_count == summary.total_count == 0


def test_retrieval_is_course_scoped_and_preserves_provenance(tmp_path: Path) -> None:
    store, _, _, _, indexing, retriever = _services(tmp_path)
    orbital = store.create_course("Orbital Gardening")
    marine = store.create_course("Marine Gardening")
    _ingest(
        store,
        orbital.id,
        tmp_path / "orbital-blueleaf.txt",
        "The Blueleaf coefficient is 7.25.",
    )
    _ingest(
        store,
        orbital.id,
        tmp_path / "solar-vines.txt",
        "Solar vines require three hours of ultraviolet exposure.",
    )
    _ingest(
        store,
        marine.id,
        tmp_path / "marine-blueleaf.txt",
        "The Blueleaf coefficient is 2.40.",
    )
    indexing.index_course(orbital.id)
    indexing.index_course(marine.id)

    orbital_results = retriever.retrieve(
        course_id=orbital.id,
        query="What is the Blueleaf coefficient?",
        limit=1,
    )
    marine_results = retriever.retrieve(
        course_id=marine.id,
        query="What is the Blueleaf coefficient?",
        limit=5,
    )
    solar_results = retriever.retrieve(
        course_id=orbital.id,
        query="How long are solar vines exposed to ultraviolet light?",
        limit=2,
    )

    assert orbital_results[0].filename == "orbital-blueleaf.txt"
    assert orbital_results[0].text == "The Blueleaf coefficient is 7.25."
    assert orbital_results[0].source_type == "section"
    assert (orbital_results[0].source_start, orbital_results[0].source_end) == (1, 1)
    assert isinstance(orbital_results[0].score, float)
    assert len(orbital_results) == 1
    assert all(result.course_id == orbital.id for result in orbital_results)
    assert all(result.course_id == marine.id for result in marine_results)
    assert solar_results[0].filename == "solar-vines.txt"


def test_unknown_course_fails_before_retrieval(tmp_path: Path) -> None:
    _, _, _, _, _, retriever = _services(tmp_path)

    with pytest.raises(CourseNotFoundError):
        retriever.retrieve(course_id="missing", query="blueleaf", limit=5)


def test_invalid_limit_fails_before_embedding(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    course = store.create_course("Boundary Course")

    class CountingProvider(DeterministicEmbeddingProvider):
        calls = 0

        def embed_query(self, text: str) -> list[float]:
            self.calls += 1
            return super().embed_query(text)

    provider = CountingProvider(dimension=64)
    index = QdrantVectorIndex(
        QdrantClient(location=":memory:"), "course_rag_test", provider
    )
    retriever = Retriever(store, index, provider)

    for limit in (0, 101):
        with pytest.raises(ValueError, match="between 1 and 100"):
            retriever.retrieve(course_id=course.id, query="blueleaf", limit=limit)

    assert provider.calls == 0


def test_retriever_rejects_provider_for_different_model(tmp_path: Path) -> None:
    store, provider, _, index, _, _ = _services(tmp_path)

    class OtherModelProvider(DeterministicEmbeddingProvider):
        model_name = "other-model"

    with pytest.raises(IndexCompatibilityError, match="does not match"):
        Retriever(store, index, OtherModelProvider(provider.dimension))


def test_existing_collection_with_wrong_metadata_is_rejected(tmp_path: Path) -> None:
    _, provider, client, index, _, _ = _services(tmp_path)
    client.create_collection(
        collection_name=index.collection_name,
        vectors_config=models.VectorParams(
            size=provider.dimension, distance=models.Distance.COSINE
        ),
        metadata={"embedding_model": "wrong"},
    )

    with pytest.raises(IndexCompatibilityError, match="incompatible"):
        index.ensure_collection()


def test_embedding_identity_selects_a_different_collection(tmp_path: Path) -> None:
    client = QdrantClient(location=":memory:")
    small = QdrantVectorIndex(
        client, "course_rag_test", DeterministicEmbeddingProvider(dimension=32)
    )
    large = QdrantVectorIndex(
        client, "course_rag_test", DeterministicEmbeddingProvider(dimension=64)
    )

    assert small.collection_name != large.collection_name


def test_missing_sqlite_chunk_fails_loudly(tmp_path: Path) -> None:
    store, provider, client, index, _, retriever = _services(tmp_path)
    course = store.create_course("Consistency Course")
    index.ensure_collection()
    missing_id = "79e6f9b8-5ae8-4b7e-92ae-98c6327bb52d"
    client.upsert(
        index.collection_name,
        points=[
            models.PointStruct(
                id=missing_id,
                vector=provider.embed_query("ghost blueleaf"),
                payload={
                    "chunk_id": missing_id,
                    "course_id": course.id,
                    "document_id": "missing-document",
                    "chunk_index": 0,
                    "source_type": "section",
                    "source_start": 1,
                    "source_end": 1,
                    "filename": "missing.txt",
                },
            )
        ],
    )

    with pytest.raises(IndexConsistencyError, match="missing from SQLite"):
        retriever.retrieve(course_id=course.id, query="ghost blueleaf", limit=5)


def test_vector_search_orders_equal_scores_by_chunk_id(tmp_path: Path) -> None:
    store, provider, client, index, _, _ = _services(tmp_path)
    course = store.create_course("Tie Course")
    first_document = _ingest(
        store, course.id, tmp_path / "first.txt", "First shared term."
    )
    second_document = _ingest(
        store, course.id, tmp_path / "second.txt", "Second shared term."
    )
    chunks = store.list_chunks(course.id)
    vector = provider.embed_query("shared term")
    index.ensure_collection()
    client.upsert(
        index.collection_name,
        points=[
            models.PointStruct(
                id=chunk.id,
                vector=vector,
                payload={
                    "chunk_id": chunk.id,
                    "course_id": course.id,
                    "document_id": first_document
                    if chunk.document_id == first_document
                    else second_document,
                    "chunk_index": chunk.chunk_index,
                    "source_type": chunk.source_type,
                    "source_start": chunk.source_start,
                    "source_end": chunk.source_end,
                    "filename": "first.txt"
                    if chunk.document_id == first_document
                    else "second.txt",
                },
            )
            for chunk in chunks
        ],
    )

    matches = index.search(course_id=course.id, vector=vector, limit=2)

    assert [match.chunk_id for match in matches] == sorted(
        chunk.id for chunk in chunks
    )
