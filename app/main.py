from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import OptimizeRequest, OptimizeResponse, Recommendation
from .services.optimizer import plan_routes, to_api_response, warmup


def create_app() -> FastAPI:
    app = FastAPI(title="Logistics Optimizer API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _startup():
        warmup()

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/metadata")
    def metadata():
        return {
            "version": app.version,
            "datasets_dir": os.path.abspath("datasets"),
            "models_dir": os.path.abspath("models"),
        }

    @app.post("/optimize", response_model=OptimizeResponse)
    def optimize(req: OptimizeRequest):
        # basic validation
        if req.preferences.max_budget is not None and req.preferences.max_budget <= 0:
            raise HTTPException(status_code=400, detail="max_budget must be > 0")

        result = plan_routes(
            origin_id=req.shipment.origin_id,
            destination_id=req.shipment.destination_id,
            weight_kg=req.shipment.weight_kg,
            volume_m3=req.shipment.volume_m3,
            cargo_class=req.shipment.cargo_class,
            required_delivery=req.shipment.required_delivery,
            optimize_for=req.preferences.optimize_for,
            allow_multimodal=req.preferences.allow_multimodal,
            k_paths=req.preferences.k_paths,
            max_budget=req.preferences.max_budget,
            scenarios=req.preferences.scenarios,
        )
        if req.preferences.max_budget is not None and not result.get("routes"):
            cheapest = result.get("cheapest_cost")
            budget = result.get("budget")
            detail = {
                "code": "over_budget",
                "message": (
                    f"No feasible route under the given budget. Cheapest is {cheapest:.0f} > budget {budget:.0f}"
                    if (cheapest is not None and budget is not None)
                    else "No feasible route under the given budget."
                ),
                "cheapest": cheapest,
                "budget": budget,
            }
            raise HTTPException(status_code=422, detail=detail)
        # If no route found (even without budget), return explicit 422 with code "no_path"
        if req.preferences.max_budget is None and not result.get("routes"):
            detail = {
                "code": "no_path",
                "message": "No feasible path between the given nodes under current constraints.",
            }
            raise HTTPException(status_code=422, detail=detail)
        api_payload = to_api_response(result, alternative_limit=1)
        # Pydantic model will generate UUIDs for route_id
        return OptimizeResponse(**api_payload)

    return app


app = create_app()
