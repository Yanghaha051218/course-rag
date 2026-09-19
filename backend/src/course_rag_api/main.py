from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, StringConstraints
from qdrant_client import QdrantClient

from course_rag_api.config import get_settings
from course_rag_api.auth import (
    authenticate_request,
    hash_password,
    issue_session,
    request_session_token,
    session_digest,
    verify_password,
)
from course_rag_api.embeddings import create_embedding_provider
from course_rag_api.errors import CourseRAGError
from course_rag_api.generation import AnswerGenerationService, create_generator
from course_rag_api.indexing import IndexingService
from course_rag_api.ingestion import ingest_document
from course_rag_api.parsers import SUPPORTED_EXTENSIONS
from course_rag_api.retrieval import Retriever
from course_rag_api.models import User
from course_rag_api.storage import SQLiteStore
from course_rag_api.support import SupportVerificationService, create_support_verifier
from course_rag_api.vector_store import QdrantVectorIndex


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    environment: str


CourseName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Email = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=3,
        max_length=254,
        pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
    ),
]
Password = Annotated[str, StringConstraints(min_length=8, max_length=256)]


class AuthRequest(BaseModel):
    email: Email
    password: Password


class UserResponse(BaseModel):
    id: str
    email: str
    created_at: str


class CreateCourseRequest(BaseModel):
    name: CourseName


class CourseResponse(BaseModel):
    id: str
    name: str
    created_at: str


class CourseListResponse(BaseModel):
    items: list[CourseResponse]


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    source_units: int
    chunks: int
    size_bytes: int


class DocumentListItem(BaseModel):
    id: str
    filename: str
    file_type: str
    source_units: int
    size_bytes: int
    created_at: str


class DocumentListResponse(BaseModel):
    items: list[DocumentListItem]


class QuestionRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class CitationResponse(BaseModel):
    chunk_id: str
    filename: str
    source_type: Literal["page", "slide", "section"]
    source_start: int
    source_end: int


class AnswerResponse(BaseModel):
    status: Literal["ANSWERED", "ABSTAINED"]
    answer: str | None
    citations: list[CitationResponse]
    reason: str | None


class EvidenceItemResponse(BaseModel):
    rank: int
    chunk_id: str
    filename: str
    text: str
    score: float
    source_type: Literal["page", "slide", "section"]
    source_start: int
    source_end: int


class EvidenceResponse(BaseModel):
    question: str
    items: list[EvidenceItemResponse]


settings = get_settings()
app = FastAPI(title=settings.app_name)


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        settings.session_cookie_name,
        token,
        max_age=settings.session_ttl_seconds,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite="lax",
        path="/",
    )


def document_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def course_storage_bytes(store: SQLiteStore, course_id: str) -> int:
    return sum(document_size(Path(document.source_path)) for document in store.list_documents(course_id))


def remove_uploaded_file(path: Path) -> None:
    try:
        path.resolve().relative_to(settings.upload_path.resolve())
    except ValueError:
        return
    path.unlink(missing_ok=True)


def get_store() -> SQLiteStore:
    return SQLiteStore(settings.database_path)


def get_current_user(
    request: Request,
    store: Annotated[SQLiteStore, Depends(get_store)],
) -> User:
    return authenticate_request(request, store, settings.session_cookie_name)


def get_indexer(
    store: Annotated[SQLiteStore, Depends(get_store)],
    _user: Annotated[User, Depends(get_current_user)],
) -> IndexingService:
    provider = create_embedding_provider(settings)
    index = QdrantVectorIndex(
        QdrantClient(path=str(settings.qdrant_path)),
        settings.qdrant_collection_prefix,
        provider,
    )
    return IndexingService(store, index, provider, batch_size=settings.embedding_batch_size)


def get_retriever(
    store: Annotated[SQLiteStore, Depends(get_store)],
    _user: Annotated[User, Depends(get_current_user)],
) -> Retriever:
    provider = create_embedding_provider(settings)
    index = QdrantVectorIndex(
        QdrantClient(path=str(settings.qdrant_path)),
        settings.qdrant_collection_prefix,
        provider,
    )
    return Retriever(store, index, provider)


def get_answer_service(
    store: Annotated[SQLiteStore, Depends(get_store)],
    _user: Annotated[User, Depends(get_current_user)],
) -> AnswerGenerationService:
    provider = create_embedding_provider(settings)
    index = QdrantVectorIndex(
        QdrantClient(path=str(settings.qdrant_path)),
        settings.qdrant_collection_prefix,
        provider,
    )
    retriever = Retriever(store, index, provider)
    support = SupportVerificationService(store, retriever, create_support_verifier(settings), 5)
    return AnswerGenerationService(support, create_generator(settings))


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Report service health",
    tags=["system"],
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        environment=settings.environment,
    )


@app.post(
    "/auth/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["auth"],
)
def register(
    request: AuthRequest,
    response: Response,
    store: Annotated[SQLiteStore, Depends(get_store)],
) -> UserResponse:
    try:
        user = store.create_user(request.email, hash_password(request.password))
    except ValueError as error:
        raise HTTPException(status_code=409, detail="Email is already registered") from error
    set_session_cookie(response, issue_session(store, user.id, settings.session_ttl_seconds))
    return UserResponse.model_validate(user, from_attributes=True)


@app.post("/auth/login", response_model=UserResponse, tags=["auth"])
def login(
    request: AuthRequest,
    response: Response,
    store: Annotated[SQLiteStore, Depends(get_store)],
) -> UserResponse:
    credentials = store.get_user_credentials(request.email)
    if credentials is None or not verify_password(request.password, credentials[1]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    user = credentials[0]
    set_session_cookie(response, issue_session(store, user.id, settings.session_ttl_seconds))
    return UserResponse.model_validate(user, from_attributes=True)


@app.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT, tags=["auth"])
def logout(
    request: Request,
    response: Response,
    store: Annotated[SQLiteStore, Depends(get_store)],
) -> Response:
    token = request_session_token(request, settings.session_cookie_name)
    if token:
        store.delete_session(session_digest(token))
    response.delete_cookie(settings.session_cookie_name, path="/")
    response.status_code = status.HTTP_204_NO_CONTENT
    return response


@app.get("/auth/me", response_model=UserResponse, tags=["auth"])
def current_user(user: Annotated[User, Depends(get_current_user)]) -> UserResponse:
    return UserResponse.model_validate(user, from_attributes=True)


@app.get("/courses", response_model=CourseListResponse, tags=["courses"])
def list_courses(
    store: Annotated[SQLiteStore, Depends(get_store)],
    user: Annotated[User, Depends(get_current_user)],
) -> CourseListResponse:
    return CourseListResponse(
        items=[
            CourseResponse.model_validate(course, from_attributes=True)
            for course in store.list_courses_for_user(user.id)
        ]
    )


@app.post(
    "/courses",
    response_model=CourseResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["courses"],
)
def create_course(
    request: CreateCourseRequest,
    store: Annotated[SQLiteStore, Depends(get_store)],
    user: Annotated[User, Depends(get_current_user)],
) -> CourseResponse:
    return CourseResponse.model_validate(
        store.create_course(request.name, owner_id=user.id), from_attributes=True
    )


@app.get(
    "/courses/{course_id}/documents",
    response_model=DocumentListResponse,
    tags=["documents"],
)
def list_documents(
    course_id: str,
    store: Annotated[SQLiteStore, Depends(get_store)],
    user: Annotated[User, Depends(get_current_user)],
) -> DocumentListResponse:
    if store.get_course_for_user(course_id, user.id) is None:
        raise HTTPException(status_code=404, detail="Course does not exist")
    return DocumentListResponse(
        items=[
            DocumentListItem(
                id=document.id,
                filename=document.filename,
                file_type=document.file_type,
                source_units=document.page_or_unit_count,
                size_bytes=document_size(Path(document.source_path)),
                created_at=document.created_at,
            )
            for document in store.list_documents(course_id)
        ]
    )


@app.post(
    "/courses/{course_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
async def upload_document(
    course_id: str,
    request: Request,
    indexer: Annotated[IndexingService, Depends(get_indexer)],
    store: Annotated[SQLiteStore, Depends(get_store)],
    user: Annotated[User, Depends(get_current_user)],
    filename: Annotated[str, Query(min_length=1, max_length=255)],
) -> DocumentResponse:
    clean_name = Path(filename).name
    if clean_name != filename or Path(clean_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=422, detail="Unsupported or unsafe filename")
    if store.get_course_for_user(course_id, user.id) is None:
        raise HTTPException(status_code=404, detail="Course does not exist")
    course_bytes = course_storage_bytes(store, course_id)
    if course_bytes >= settings.max_course_bytes:
        raise HTTPException(status_code=413, detail="Course storage quota exceeded")
    upload_dir = settings.upload_path / course_id / str(uuid4())
    upload_dir.mkdir(parents=True, exist_ok=False)
    path = upload_dir / clean_name
    size = 0
    try:
        with path.open("xb") as target:
            async for block in request.stream():
                size += len(block)
                if size > settings.max_document_bytes:
                    raise HTTPException(status_code=413, detail="Document is too large")
                if course_bytes + size > settings.max_course_bytes:
                    raise HTTPException(status_code=413, detail="Course storage quota exceeded")
                target.write(block)
        summary = ingest_document(
            store=store,
            course_id=course_id,
            file_path=path,
            chunk_target_size=settings.chunk_target_size,
            chunk_overlap=settings.chunk_overlap,
            max_document_bytes=settings.max_document_bytes,
        )
        indexer.index_document(summary.document_id)
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    except (CourseRAGError, ValueError) as error:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(error)) from error
    return DocumentResponse(
        id=summary.document_id,
        filename=summary.filename,
        file_type=summary.file_type,
        source_units=summary.source_unit_count,
        chunks=summary.chunk_count,
        size_bytes=size,
    )


@app.delete(
    "/courses/{course_id}/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["documents"],
)
def delete_document(
    course_id: str,
    document_id: str,
    indexer: Annotated[IndexingService, Depends(get_indexer)],
    store: Annotated[SQLiteStore, Depends(get_store)],
    user: Annotated[User, Depends(get_current_user)],
) -> Response:
    try:
        if store.get_course_for_user(course_id, user.id) is None:
            raise ValueError("course does not belong to user")
        document = store.get_document(document_id)
        if document.course_id != course_id:
            raise ValueError("document does not belong to course")
        indexer.delete_document(course_id, document_id)
        remove_uploaded_file(Path(document.source_path))
    except (CourseRAGError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Document does not exist") from error
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post(
    "/courses/{course_id}/questions",
    response_model=AnswerResponse,
    tags=["questions"],
)
def answer_question(
    course_id: str,
    request: QuestionRequest,
    store: Annotated[SQLiteStore, Depends(get_store)],
    service: Annotated[AnswerGenerationService, Depends(get_answer_service)],
    user: Annotated[User, Depends(get_current_user)],
) -> AnswerResponse:
    try:
        if store.get_course_for_user(course_id, user.id) is None:
            raise HTTPException(status_code=404, detail="Course does not exist")
        result = service.answer_question(course_id=course_id, question=request.question)
    except HTTPException:
        raise
    except CourseRAGError as error:
        raise HTTPException(status_code=503, detail="Answer service unavailable") from error
    return AnswerResponse(
        status=result.status.value,
        answer=result.answer,
        citations=[CitationResponse.model_validate(citation, from_attributes=True) for citation in result.citations],
        reason=result.reason,
    )


@app.post(
    "/courses/{course_id}/evidence",
    response_model=EvidenceResponse,
    tags=["questions"],
)
def retrieve_evidence(
    course_id: str,
    request: QuestionRequest,
    retriever: Annotated[Retriever, Depends(get_retriever)],
    store: Annotated[SQLiteStore, Depends(get_store)],
    user: Annotated[User, Depends(get_current_user)],
) -> EvidenceResponse:
    if store.get_course_for_user(course_id, user.id) is None:
        raise HTTPException(status_code=404, detail="Course does not exist")
    try:
        chunks = retriever.retrieve(course_id=course_id, query=request.question, limit=5)
    except CourseRAGError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return EvidenceResponse(
        question=request.question,
        items=[
            EvidenceItemResponse(
                rank=rank,
                chunk_id=chunk.chunk_id,
                filename=chunk.filename,
                text=chunk.text,
                score=chunk.score,
                source_type=chunk.source_type,
                source_start=chunk.source_start,
                source_end=chunk.source_end,
            )
            for rank, chunk in enumerate(chunks, 1)
        ],
    )
