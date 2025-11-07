import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi.testclient import TestClient
from app.main import create_app
import json

client = TestClient(create_app())

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
print("status_code:", resp.status_code)
try:
    print(json.dumps(resp.json(), indent=2))
except Exception:
    print(resp.text)
