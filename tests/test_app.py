from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data


def test_predict_invalid_file():
    # Sending dummy text file instead of image should fail gracefully
    files = {"file": ("test.txt", b"dummy content", "text/plain")}
    response = client.post("/predict?model=cnn", files=files)
    assert response.status_code == 400
    assert "error" in response.json()
