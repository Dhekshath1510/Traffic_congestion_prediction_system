from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_liveness_check() -> None:
    """Verify that the liveness endpoint responds successfully."""
    response = client.get("/api/v1/health/live")
    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "Traffic Congestion Prediction API"
    assert data["version"] == "0.1.0"
