import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.optimizer import plan_routes
import json
from datetime import datetime

try:
    res = plan_routes(
        origin_id="NODE_000",
        destination_id="NODE_010",
        weight_kg=1200,
        volume_m3=12,
        cargo_class="standard",
        required_delivery=datetime.fromisoformat("2024-01-05T12:00:00+00:00"),
        optimize_for="balanced",
        allow_multimodal=True,
        k_paths=1,
        scenarios=10,
        max_budget=120000,
    )
    print("plan_routes returned keys:", list(res.keys()))
    print(json.dumps({k: res[k] for k in res if k != 'routes'}, indent=2))
    print("routes count:", len(res.get('routes', [])))
except Exception as e:
    import traceback

    print("Exception in plan_routes:")
    traceback.print_exc()
