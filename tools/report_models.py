import os
import json
from typing import Dict, Any

import numpy as np
import pandas as pd


def _safe_read_json(path: str) -> Dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _exists(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) > 0
    except Exception:
        return False


def _fmt(v: float, digits: int = 3) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return "nan"


def report_demand() -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": False}
    m = _safe_read_json("models/demand_forecast_metrics.json")
    if m and isinstance(m, dict):
        out.update(m)
    # recompute MAPE if predictions exist
    try:
        if _exists("models/demand_forecast_predictions.csv"):
            df = pd.read_csv("models/demand_forecast_predictions.csv")
            if {"y_true", "y_pred"}.issubset(df.columns):
                y_true = df["y_true"].to_numpy(dtype=float)
                y_pred = df["y_pred"].to_numpy(dtype=float)
                denom = np.clip(np.abs(y_true), 1e-6, None)
                mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
                out["mape_pred_file"] = mape
    except Exception:
        pass
    # health heuristic
    mape_val = out.get("mape") or out.get("mape_pred_file")
    if isinstance(mape_val, (int, float)):
        if mape_val < 20:
            out["health"] = "good"
        elif mape_val < 40:
            out["health"] = "warn"
        else:
            out["health"] = "poor"
    out["ok"] = True if m else False
    return out


def report_transport() -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": False}
    m = _safe_read_json("models/transport_classifier_metrics.json")
    if m:
        out.update({
            "accuracy": m.get("accuracy"),
            "macro_f1": m.get("macro_f1"),
            "n_train": m.get("n_train"),
            "n_test": m.get("n_test"),
        })
        # health heuristic
        acc = out.get("accuracy")
        f1 = out.get("macro_f1")
        if isinstance(acc, (int, float)) and isinstance(f1, (int, float)):
            if acc >= 0.75 and f1 >= 0.7:
                out["health"] = "good"
            elif acc >= 0.6 and f1 >= 0.55:
                out["health"] = "warn"
            else:
                out["health"] = "poor"
        out["ok"] = True
    return out


def report_l2r() -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": False}
    m = _safe_read_json("models/learn_to_route_metrics.json")
    if m:
        out.update(m)
        # attempt heuristic if MAE present
        mae_t = out.get("time_mae") or out.get("mae_time")
        mae_c = out.get("cost_mae") or out.get("mae_cost")
        if isinstance(mae_t, (int, float)):
            out["health_time"] = "good" if mae_t < 5 else ("warn" if mae_t < 12 else "poor")
        if isinstance(mae_c, (int, float)):
            out["health_cost"] = "good" if mae_c < 500 else ("warn" if mae_c < 1500 else "poor")
        out["ok"] = True
    return out


def report_gnn() -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": False}
    m = _safe_read_json("models/edge_gnn_metrics.json")
    if m:
        out.update(m)
        out["ok"] = True
        return out
    # Try a lightweight evaluation via built-in inferencer if available
    try:
        from ml.gnn_models import EdgeGNNInferencer  # type: ignore
        if os.path.exists("models/edge_gnn.pt"):
            inf = EdgeGNNInferencer.load("models/edge_gnn.pt")
            if hasattr(inf, "metrics"):
                out.update({"metrics": getattr(inf, "metrics")})
            out["present"] = True
            out["ok"] = True
    except Exception:
        out["present"] = os.path.exists("models/edge_gnn.pt")
    return out


def report_rl() -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": False}
    out["present"] = os.path.exists("models/rl_policy.pt")
    out["ok"] = True
    return out


def main():
    sections = {
        "demand_forecast": report_demand(),
        "transport_classifier": report_transport(),
        "learn_to_route": report_l2r(),
        "edge_gnn": report_gnn(),
        "rl_policy": report_rl(),
    }
    print("\nModel Quality Report:\n----------------------")
    for name, s in sections.items():
        print(f"\n[{name}]")
        if not s.get("ok"):
            print("  metrics: not found")
            continue
        for k, v in s.items():
            if k == "ok":
                continue
            if isinstance(v, float):
                print(f"  {k}: {_fmt(v)}")
            else:
                print(f"  {k}: {v}")

    # quick overall verdict
    verdicts = []
    d = sections.get("demand_forecast", {})
    if d.get("health") == "good": verdicts.append("demand ok")
    t = sections.get("transport_classifier", {})
    if t.get("health") == "good": verdicts.append("transport ok")
    l = sections.get("learn_to_route", {})
    if l.get("health_time") == "good" and l.get("health_cost") == "good": verdicts.append("l2r ok")
    print("\nVerdict:")
    if verdicts:
        print("  ", ", ".join(verdicts))
    else:
        print("  more training/metrics needed to judge")


if __name__ == "__main__":
    main()
