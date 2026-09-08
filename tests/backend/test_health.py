from fastapi.testclient import TestClient

from course_rag_api.main import app


def test_health_returns_service_status() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "CourseRAG API",
        "environment": "development",
    }
