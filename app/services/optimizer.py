from __future__ import annotations

from typing import Dict, List
from datetime import datetime

from ml.route_optimizer import optimize_delivery
from ml import route_optimizer as ro
from .scoring import weights_from_preference, confidence_from_scores


def _adjust_weights_for_cargo(weights: Dict[str, float], cargo_class: str) -> Dict[str, float]:
    # Lightweight heuristic: emphasize time for express, reliability for fragile, cost for standard
    w = dict(weights)
    cc = (cargo_class or "standard").lower()
    if cc == "express":
        w["w_time"] = w.get("w_time", 1.0) * 1.6
    elif cc == "fragile":
        w["w_rel"] = w.get("w_rel", 1.0) * 1.5
    else:  # standard
        w["w_cost"] = w.get("w_cost", 1.0) * 1.2
    return w


def plan_routes(
    origin_id: str,
    destination_id: str,
    weight_kg: float,
    volume_m3: float,
    cargo_class: str,
    required_delivery: datetime,
    optimize_for: str = "balanced",
    allow_multimodal: bool = True,
    k_paths: int = 3,
    scenarios: int = 1,
    max_budget: int | None = None,
) -> Dict:
    # map preferences to optimizer args
    weights = weights_from_preference(optimize_for)
    # adjust by cargo class
    weights = _adjust_weights_for_cargo(weights, cargo_class)
    mode_pref = "combined" if allow_multimodal else "road"
    cargo_specs = {
        "weight_kg": weight_kg,
        "volume_m3": volume_m3,
        "mode_pref": mode_pref,
        "created_at": required_delivery,
    }
    # Simple LRU cache lookup for nominal call
    key = _cache_key(origin_id, destination_id, weight_kg, volume_m3, mode_pref, required_delivery, weights)
    cached = _cache_get(key)
    if cached is not None:
        all_routes = cached
    else:
        all_routes = optimize_delivery(
            origin=origin_id,
            destination=destination_id,
            cargo_specs=cargo_specs,
            constraints=weights,
            k_paths=k_paths,
        )
        _cache_set(key, all_routes)
    # Optional budget filter
    routes = list(all_routes)
    if max_budget is not None:
        routes = [r for r in routes if float(r.get("cost", 0.0)) <= float(max_budget)]
    # Keep only the best route for API contract (single recommendation)
    top_only = routes[:1] if routes else []
    # Risk-aware scenarios (Monte Carlo on per-segment time/cost)
    risk = None
    if top_only and scenarios and scenarios > 1:
        r = top_only[0]
        times = []
        costs = []
        for _ in range(int(scenarios)):
            t_sum = 0.0
            c_sum = 0.0
            prev_mode = None
            for s in r.get("segments", []):
                mode = s.get("mode", "road")
                base_t = float(s.get("pred_time_h", s.get("base_time_h", 0.0)))
                base_c = float(s.get("pred_cost", s.get("base_cost", 0.0)))
                # mode-dependent noise
                if mode == "air":
                    t_std, c_std = 0.05, 0.08
                else:  # road/combined
                    t_std, c_std = 0.07, 0.10
                t = random.gauss(base_t, base_t * t_std)
                c = random.gauss(base_c, base_c * c_std)
                t_sum += max(0.0, t)
                c_sum += max(0.0, c)
            times.append(t_sum)
            costs.append(c_sum)
        import statistics as st
        risk = {
            "expected_time_h": float(st.mean(times)),
            "expected_cost": float(st.mean(costs)),
            "time_std_h": float(st.pstdev(times)) if len(times) > 1 else 0.0,
            "cost_std": float(st.pstdev(costs)) if len(costs) > 1 else 0.0,
        }
    cheapest_cost = None
    if all_routes:
        cheapest_cost = float(min(float(r.get("cost", 0.0)) for r in all_routes))
    return {
        "routes": top_only,
        "weights": weights,
        "cheapest_cost": cheapest_cost,
        "budget": max_budget,
        "risk": risk,
        "scenarios": int(scenarios or 1),
    }


def to_api_response(planned: Dict, alternative_limit: int = 1) -> Dict:
    routes = planned.get("routes", [])

    recs = []
    if routes:
        r = routes[0]
        segs = []
        allowed_modes = {"road", "air", "combined"}
        norm_modes = []
        for s in r.get("segments", []):
            raw_mode = s.get("mode", "road")
            mode = raw_mode if raw_mode in allowed_modes else "combined"
            norm_modes.append(mode)
            segs.append(
                {
                    "mode": mode,
                    "from": s.get("from_node"),
                    "to": s.get("to_node"),
                    "duration_hours": float(s.get("pred_time_h", s.get("base_time_h", 0.0))),
                    "cost": float(s.get("pred_cost", s.get("base_cost", 0.0))),
                }
            )
        # Use route reliability as confidence proxy for a single route
        conf = float(max(0.0, min(1.0, r.get("reliability", 0.0))))
        # Cost breakdown by mode (use normalized segment entries)
        by_mode = {}
        for seg in segs:
            m = seg.get("mode", "road")
            c = float(seg.get("cost", 0.0))
            by_mode[m] = by_mode.get(m, 0.0) + c

        # Risk factors
        risk_factors = []
        switches = sum(1 for i in range(1, len(norm_modes)) if norm_modes[i] != norm_modes[i-1])
        if switches > 0:
            risk_factors.append("mode_switches")
        if float(r.get("reliability", 1.0)) < 0.85:
            risk_factors.append("low_reliability")

        # remove duplicates while preserving order for transport modes; use normalized modes
        transport_modes = list(dict.fromkeys(norm_modes))
        recs.append(
            {
                "transport_modes": transport_modes,
                "segments": segs,
                "total_cost": float(r.get("cost", 0.0)),
                "total_duration_hours": float(r.get("time_h", 0.0)),
                "reliability_score": float(r.get("reliability", 0.0)),
                "ml_confidence": conf,
            }
        )

    analytics = {
        "cost_breakdown": by_mode if routes else {},
        "risk_factors": risk_factors if routes else [],
        "alternative_routes": 1 if routes else 0,
    }
    # Attach risk summary if computed
    if planned.get("risk"):
        analytics.update(planned["risk"])  # expected_time_h, expected_cost, stds
    # Also attach scenarios count
    analytics["scenarios"] = planned.get("scenarios", 1)

    return {
        "recommendations": recs,
        "analytics": analytics,
    }
from collections import OrderedDict
import random

# Simple in-process LRU cache for route results
_ROUTE_CACHE: OrderedDict[str, dict] = OrderedDict()
_ROUTE_CACHE_CAP = 128


def warmup() -> None:
    """Preload graph/models to warm OS caches and lazy imports."""
    try:
        # Touch internal loaders (best-effort)
        _ = ro._load_graph()
        _ = ro._load_edge_models()
    except Exception:
        pass


def _cache_key(origin: str, dest: str, weight: float, volume: float, mode_pref: str, created_at, weights: dict) -> str:
    return f"{origin}|{dest}|{weight:.1f}|{volume:.1f}|{mode_pref}|{str(getattr(created_at,'date',lambda:created_at)())}|{weights}"


def _cache_get(key: str):
    v = _ROUTE_CACHE.get(key)
    if v is not None:
        # move to end (recent)
        _ROUTE_CACHE.move_to_end(key)
    return v


def _cache_set(key: str, value: dict) -> None:
    _ROUTE_CACHE[key] = value
    _ROUTE_CACHE.move_to_end(key)
    if len(_ROUTE_CACHE) > _ROUTE_CACHE_CAP:
        _ROUTE_CACHE.popitem(last=False)
