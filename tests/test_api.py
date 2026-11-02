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
        "generated_text": "The Earth revolves around the Sun.",
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
    assert data["status"] in ("PASSED", "PASSED_MOCK")
    assert data["trust_score"] >= 0.7
    assert data["action"] == "PASS"

def test_invalid_payload_fails():
    # Sending payload with missing 'generated_text'
    payload = {
        "request_id": "test_456",
        "retrieved_contexts": []
    }
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 422  # Validation Error

def test_v1_health_and_config():
    h_res = client.get("/api/v1/health")
    assert h_res.status_code == 200
    assert h_res.json()["status"] == "ok"

    c_res = client.get("/api/v1/config")
    assert c_res.status_code == 200
    assert "embedding_model" in c_res.json()

def test_v1_evaluate_contradiction_blocking():
    payload = {
        "request_id": "test_con",
        "generated_text": "Company profit decreased by 50 percent in 2024.",
        "retrieved_contexts": [
            {
                "doc_id": "doc_fin",
                "text": "The audited annual report confirms company profit increased by 50 percent in 2024."
            }
        ]
    }
    response = client.post("/api/v1/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["action"] == "BLOCK"
    assert data["status"] == "BLOCKED"
    assert len(data["flagged_claims"]) >= 1

def test_v1_evaluate_batch():
    batch_payload = [
        {
            "request_id": "req_batch_1",
            "generated_text": "Jupiter is a gas giant planet.",
            "retrieved_contexts": [{"doc_id": "d1", "text": "Jupiter is a gas giant and the largest planet."}]
        },
        {
            "request_id": "req_batch_2",
            "generated_text": "Mars is known as the Red Planet.",
            "retrieved_contexts": [{"doc_id": "d2", "text": "Mars is widely recognized as the Red Planet."}]
        }
    ]
    response = client.post("/api/v1/evaluate/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["action"] == "PASS"
    assert data[1]["action"] == "PASS"

def test_filtering_hallucinations():
    payload = {
        "request_id": "test_filter",
        "generated_text": "The Earth orbits the Sun. Unicorns live on Jupiter.",
        "retrieved_contexts": [
            {
                "doc_id": "d1",
                "text": "The Earth orbits the Sun once every year."
            }
        ],
        "filter_hallucinations": True
    }
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "The Earth orbits the Sun" in (data["filtered_text"] or "")
    assert "Unicorns live on Jupiter" not in (data["filtered_text"] or "")