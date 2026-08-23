from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "TrustLayer"}

def test_valid_evaluation_request():
    payload = {
        "request_id": "test_123",
        "generated_text": "The earth orbits the sun.",
        "retrieved_contexts": [
            {
                "doc_id": "doc_1",
                "text": "Astronomical data confirms the Earth revolves around the Sun."
            }
        ]
    }
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "PASSED_MOCK"
    assert data["trust_score"] == 1.0

def test_invalid_payload_fails():
    # Sending payload with missing 'generated_text'
    payload = {
        "request_id": "test_456",
        "retrieved_contexts": []
    }
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 422  # Validation Error