from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from course_rag_api.config import get_settings


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    environment: str


settings = get_settings()
app = FastAPI(title=settings.app_name)


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
