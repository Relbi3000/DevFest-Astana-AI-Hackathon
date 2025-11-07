from __future__ import annotations

from typing import Dict, List


def weights_from_preference(pref: str) -> Dict[str, float]:
    pref = (pref or "balanced").lower()
    if pref == "time":
        return {"w_time": 2.0, "w_cost": 1.0, "w_rel": 1.0}
    if pref == "cost":
        return {"w_time": 1.0, "w_cost": 2.0, "w_rel": 1.0}
    return {"w_time": 1.0, "w_cost": 1.0, "w_rel": 1.0}


def confidence_from_scores(scores: List[float]) -> float:
    if not scores:
        return 0.0
    top = max(scores)
    if len(scores) == 1:
        return 1.0
    second = sorted(scores, reverse=True)[1]
    # margin normalized by top score
    if top <= 0:
        return 0.0
    margin = (top - second) / top
    # clamp to [0,1]
    return float(max(0.0, min(1.0, margin)))

