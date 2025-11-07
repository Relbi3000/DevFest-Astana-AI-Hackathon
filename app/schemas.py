from __future__ import annotations

from typing import List, Literal, Optional
from uuid import UUID, uuid4
from datetime import datetime

from pydantic import BaseModel, Field, confloat, conint, validator


class Shipment(BaseModel):
    origin_id: str = Field(..., description="Start node id")
    destination_id: str = Field(..., description="End node id")
    weight_kg: confloat(gt=0) = Field(...)
    volume_m3: confloat(gt=0) = Field(...)
    cargo_class: Literal["standard", "express", "fragile"]
    required_delivery: datetime


class Preferences(BaseModel):
    optimize_for: Literal["cost", "time", "balanced"] = "balanced"
    max_budget: Optional[conint(gt=0)] = None
    allow_multimodal: bool = True
    k_paths: conint(ge=1, le=5) = 3
    scenarios: conint(ge=1, le=50) = 1


class OptimizeRequest(BaseModel):
    shipment: Shipment
    preferences: Preferences = Field(default_factory=Preferences)


class Segment(BaseModel):
    mode: Literal["road", "air", "combined"]
    from_: str = Field(..., alias="from")
    to: str
    duration_hours: float
    cost: float

    class Config:
        # Pydantic v2: use populate_by_name instead of allow_population_by_field_name
        populate_by_name = True


class Recommendation(BaseModel):
    route_id: UUID = Field(default_factory=uuid4)
    transport_modes: List[Literal["road", "air", "combined"]]
    segments: List[Segment]
    total_cost: float
    total_duration_hours: float
    reliability_score: float
    ml_confidence: float


class Analytics(BaseModel):
    cost_breakdown: dict = Field(default_factory=dict)
    risk_factors: List[str] = Field(default_factory=list)
    alternative_routes: int = 0


class OptimizeResponse(BaseModel):
    recommendations: List[Recommendation]
    analytics: Analytics
