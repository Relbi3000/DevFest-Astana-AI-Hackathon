from fastapi.testclient import TestClient
from app.main import create_app


client = TestClient(create_app())


def test_optimize_request_no_500():
    payload = {
        "shipment": {
            "origin_id": "NODE_000",
            "destination_id": "NODE_010",
            "weight_kg": 1200,
            "volume_m3": 12,
            "cargo_class": "standard",
            "required_delivery": "2024-01-05T12:00:00Z",
        },
        "preferences": {
            "optimize_for": "balanced",
            "max_budget": 120000,
            "allow_multimodal": True,
            "k_paths": 1,
            "scenarios": 10,
        },
    }

    resp = client.post("/optimize", json=payload)
    # Should not return a 500. Accept 200 (OK) or 422 (no feasible route / over budget).
    assert resp.status_code in (200, 422), f"Unexpected status: {resp.status_code}, body: {resp.text}"
