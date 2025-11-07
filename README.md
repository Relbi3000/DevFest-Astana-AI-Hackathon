# Edgerunners Logistics Optimizer

End-to-end synthetic data generator, ML pipeline, and REST API for multimodal logistics routing.

## Quick Start

1. Create venv and install deps

```
python bootstrap.py
# PowerShell: .\.venv\Scripts\Activate
# Linux/macOS: source .venv/bin/activate
```

2. Generate datasets and train models

```
python run_pipeline.py
```

Artifacts are written to `datasets/` (CSV, graph) and `models/` (trained models, metrics).

3. Run API

```
python -m uvicorn app.main:app --reload
# Open http://127.0.0.1:8000/docs
```

## API

- `GET /healthz` — liveness
- `GET /metadata` — version and paths
- `POST /optimize` — route recommendation

Example request:

```
{
  "shipment": {
    "origin_id": "NODE_000",
    "destination_id": "NODE_010",
    "weight_kg": 1200,
    "volume_m3": 12,
    "cargo_class": "standard",
    "required_delivery": "2024-01-05T12:00:00Z"
  },
  "preferences": {
    "optimize_for": "balanced",
    "max_budget": 120000,
    "allow_multimodal": true,
    "k_paths": 1,
    "scenarios": 10
  }
}
```

Responses include one route (segments, total cost/time, reliability), analytics (cost breakdown, risk factors), and risk-aware summary (expected time/cost and std) when `scenarios > 1`.

Budget errors return 422 with JSON detail `{ code: "over_budget", cheapest, budget }`. If no path exists at all, `{ code: "no_path" }`.

## Docker

Build and run container:
REPLACE "<project-reository-name>"
```
docker build -t <project-repository-name>:latest .
docker run -d -p 8000:8000 <project-repository-name>
```

Open Swagger at http://localhost:8000/docs

## Postman

Import `postman/collection.json` into Postman. Set `baseUrl` to your host (default `http://localhost:8000`).

## Project Structure

- `datagenerator4.py` — synthetic data generation
- `ml/` — ML pipeline: demand forecast, transport classifier, route optimizer, learn-to-route
- `app/` — FastAPI service
- `datasets/` — generated datasets (created by pipeline)
- `models/` — trained models and metrics (created by pipeline)

## Notes

- The API warms up graph/models and caches route computations (LRU) for better latency.
- Risk-aware routing uses simple scenario noise per segment (by mode). Extend as needed.

