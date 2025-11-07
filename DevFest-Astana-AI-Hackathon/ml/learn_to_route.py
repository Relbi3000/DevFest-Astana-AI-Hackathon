"""
Learn-to-Route: train per-segment predictors for time and cost.

We approximate per-segment targets by splitting order-level actual_time_h and actual_cost
proportionally to each segment's base_time_h and base_cost within that order.

Artifacts:
- models/edge_time_model.pkl
- models/edge_cost_model.pkl
"""
import os
import json
import pickle
from typing import Tuple, Dict

import numpy as np
import pandas as pd

# Support module/script and datasets path resolution
try:
    from .utils import resolve_path, load_nodes_edges
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import resolve_path, load_nodes_edges


def _build_order_segments_from_orders(orders: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Construct order_segments from orders with route_nodes/route_modes and edges.

    Output columns: order_id, seq, from_node, to_node, mode, edge_id, distance_km, base_time_h, base_cost
    """
    # index edges by undirected pair + mode
    idx: dict[tuple[frozenset[str], str], dict] = {}
    for _, e in edges.iterrows():
        key = (frozenset([str(e["from_node"]), str(e["to_node"]) ]), str(e.get("mode", "road")))
        idx[key] = e.to_dict()

    rows: list[dict] = []
    # ensure strings
    def _split(s: str) -> list[str]:
        if pd.isna(s) or s is None:
            return []
        return [x for x in str(s).split(";") if len(x)]

    for _, r in orders.iterrows():
        oid = r.get("order_id")
        nodes = _split(r.get("route_nodes"))
        modes = _split(r.get("route_modes"))
        # default: if modes length mismatches, pad with last/road
        if len(modes) < max(len(nodes) - 1, 0):
            modes = modes + [modes[-1] if modes else "road"] * (len(nodes) - 1 - len(modes))
        for i in range(max(len(nodes) - 1, 0)):
            u = nodes[i]; v = nodes[i + 1]; m = modes[i] if i < len(modes) else "road"
            ek = (frozenset([u, v]), str(m))
            ed = idx.get(ek)
            row = {
                "order_id": oid,
                "seq": i,
                "from_node": u,
                "to_node": v,
                "mode": m,
                "edge_id": ed.get("edge_id") if ed else None,
                "distance_km": float(ed.get("distance_km", np.nan)) if ed else np.nan,
                "base_time_h": float(ed.get("base_time_h", np.nan)) if ed else np.nan,
                "base_cost": float(ed.get("base_cost", np.nan)) if ed else np.nan,
            }
            rows.append(row)
    seg = pd.DataFrame(rows)
    return seg


def _load_required():
    # orders first (robust path and dates)
    try:
        from .utils import resolve_path as _rp  # type: ignore
    except Exception:
        # fallback when running as a script
        from utils import resolve_path as _rp  # type: ignore
    orders = pd.read_csv(_rp("orders_test.csv", synonyms=["orders.csv"]))
    if "created_at" in orders.columns:
        try:
            orders["created_at"] = pd.to_datetime(orders["created_at"])
        except Exception:
            pass
    # segments: prefer file, else build
    seg_path = _rp("order_segments.csv")
    seg = None
    try:
        if os.path.exists(seg_path):
            seg = pd.read_csv(seg_path)
    except Exception:
        seg = None
    if seg is None or len(seg) == 0:
        # build from orders + edges
        nodes, edges = load_nodes_edges()
        seg = _build_order_segments_from_orders(orders, edges)
    # weather optional
    weather = None
    wp = _rp("weather.csv")
    if os.path.exists(wp):
        try:
            weather = pd.read_csv(wp, parse_dates=["date"]) if os.path.getsize(wp) > 0 else None
        except Exception:
            w = pd.read_csv(wp)
            if w is not None and "date" in w.columns:
                try:
                    w["date"] = pd.to_datetime(w["date"]) 
                except Exception:
                    pass
            weather = w
    # normalize weather factor columns
    if weather is not None and len(weather) > 0:
        if "time_factor" not in weather.columns and "time_factor_road" in weather.columns:
            weather = weather.copy()
            weather["time_factor"] = weather["time_factor_road"]
        if "cost_factor" not in weather.columns and "cost_factor_road" in weather.columns:
            weather = weather.copy()
            weather["cost_factor"] = weather["cost_factor_road"]
    return seg, orders, weather


def _prepare_training() -> Tuple[pd.DataFrame, pd.Series, pd.Series, list, pd.Series]:
    seg, orders, weather = _load_required()
    # merge capacity attributes and reliability from edges
    nodes, edges = load_nodes_edges()
    if "edge_id" in seg.columns and "edge_id" in edges.columns:
        cols = [c for c in ["edge_id", "max_weight_kg", "max_volume_m3", "reliability"] if c in edges.columns]
        seg = seg.merge(edges[cols], on="edge_id", how="left")
    # default reliability if still missing
    if "reliability" not in seg.columns:
        seg["reliability"] = 0.95
    # Aggregate base sums per order
    seg_sum = seg.groupby("order_id").agg(sum_base_time=("base_time_h", "sum"), sum_base_cost=("base_cost", "sum")).reset_index()
    df = seg.merge(seg_sum, on="order_id", how="left")
    # Join order fields
    keep = [
        "order_id", "origin_id", "destination_id", "created_at",
        "weight_kg", "volume_m3", "cargo_class", "actual_time_h", "actual_cost",
    ]
    df = df.merge(orders[keep], on="order_id", how="left")
    # Weather by date + from_node (segment origin is from_node)
    if weather is not None and len(weather) > 0:
        w = weather.copy()
        w["date"] = pd.to_datetime(w["date"]).dt.date
        df["date"] = pd.to_datetime(df["created_at"]).dt.date
        df = df.merge(
            w[["date", "node_id", "time_factor", "cost_factor"]],
            left_on=["date", "from_node"], right_on=["date", "node_id"], how="left",
        )
        df = df.drop(columns=[c for c in ["node_id"] if c in df.columns])
    else:
        df["time_factor"] = 1.0
        df["cost_factor"] = 1.0
    # Targets: proportional split
    eps = 1e-6
    df["time_target"] = df["actual_time_h"] * (df["base_time_h"] / (df["sum_base_time"] + eps))
    df["cost_target"] = df["actual_cost"] * (df["base_cost"] / (df["sum_base_cost"] + eps))
    # Features
    df = pd.get_dummies(df, columns=["cargo_class", "mode"], drop_first=False)
    feature_cols = [
        "distance_km", "base_time_h", "base_cost", "max_weight_kg", "max_volume_m3", "reliability",
        "time_factor", "cost_factor", "weight_kg", "volume_m3",
    ] + [c for c in df.columns if c.startswith("cargo_class_") or c.startswith("mode_")]
    X = df[feature_cols].fillna(0.0)
    y_time = df["time_target"].values
    y_cost = df["cost_target"].values
    created_at = pd.to_datetime(df["created_at"]) if "created_at" in df.columns else pd.Series([pd.Timestamp(0)] * len(df))
    return X, y_time, y_cost, feature_cols, created_at


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    eps = 1e-6
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    # robust MAPE: игнорируем почти нулевые таргеты
    # порог адаптивный: max(1e-3, 0.01 * median(|y|))
    thr = max(1e-3, 0.01 * float(np.median(np.abs(yt)) if len(yt) else 0.0))
    mask = np.abs(yt) >= thr
    if not np.any(mask):
        mape = float('nan')
    else:
        denom = np.clip(np.abs(yt[mask]), eps, None)
        mape = float(np.mean(np.abs((yt[mask] - yp[mask]) / denom)) * 100.0)
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    return {"mape": mape, "rmse": rmse, "mae": mae}


def train_and_save():
    os.makedirs("models", exist_ok=True)
    X, y_time, y_cost, feature_cols, created_at = _prepare_training()

    # time-based split (last 20% as test)
    try:
        order_idx = created_at.sort_values().index
        X = X.loc[order_idx].reset_index(drop=True)
        y_time = y_time[order_idx]
        y_cost = y_cost[order_idx]
    except Exception:
        pass

    n = len(X)
    split = int(n * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_time_tr, y_time_te = y_time[:split], y_time[split:]
    y_cost_tr, y_cost_te = y_cost[:split], y_cost[split:]

    model_time = None
    model_cost = None
    seed = 42
    try:
        import lightgbm as lgb
        model_time = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, max_depth=-1, random_state=seed)
        model_time.fit(X_train, y_time_tr)
        model_cost = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, max_depth=-1, random_state=seed)
        model_cost.fit(X_train, y_cost_tr)
    except Exception:
        try:
            import xgboost as xgb
            model_time = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.8, random_state=seed)
            model_time.fit(X_train, y_time_tr)
            model_cost = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.8, random_state=seed)
            model_cost.fit(X_train, y_cost_tr)
        except Exception:
            from sklearn.ensemble import RandomForestRegressor
            model_time = RandomForestRegressor(n_estimators=400, random_state=seed)
            model_time.fit(X_train, y_time_tr)
            model_cost = RandomForestRegressor(n_estimators=400, random_state=seed)
            model_cost.fit(X_train, y_cost_tr)

    # Evaluate
    y_time_pred = model_time.predict(X_test)
    y_cost_pred = model_cost.predict(X_test)
    metrics = {
        "time": _compute_metrics(y_time_te, y_time_pred),
        "cost": _compute_metrics(y_cost_te, y_cost_pred),
        "n_train": int(split),
        "n_test": int(n - split),
    }

    with open("models/edge_time_model.pkl", "wb") as f:
        pickle.dump({"model": model_time, "feature_cols": feature_cols}, f)
    with open("models/edge_cost_model.pkl", "wb") as f:
        pickle.dump({"model": model_cost, "feature_cols": feature_cols}, f)

    # Optional: quantile models for stochastic optimization
    try:
        import lightgbm as lgb  # type: ignore
        qt: Dict[str, object] = {}
        qc: Dict[str, object] = {}
        for alpha in [0.1, 0.5, 0.9]:
            mtq = lgb.LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=600, learning_rate=0.05, max_depth=-1, random_state=seed)
            mtq.fit(X_train, y_time_tr)
            qt[str(alpha)] = mtq
            mcq = lgb.LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=600, learning_rate=0.05, max_depth=-1, random_state=seed)
            mcq.fit(X_train, y_cost_tr)
            qc[str(alpha)] = mcq
        with open("models/edge_time_model_quantiles.pkl", "wb") as f:
            pickle.dump({"models": qt, "feature_cols": feature_cols}, f)
        with open("models/edge_cost_model_quantiles.pkl", "wb") as f:
            pickle.dump({"models": qc, "feature_cols": feature_cols}, f)
        metrics["quantiles_available"] = True
    except Exception:
        metrics["quantiles_available"] = False

    with open("models/learn_to_route_config.json", "w", encoding="utf-8") as f:
        json.dump({"feature_cols": feature_cols, "seed": seed}, f, ensure_ascii=False, indent=2)
    with open("models/learn_to_route_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("Saved edge predictors: models/edge_time_model.pkl, models/edge_cost_model.pkl")


def main():
    train_and_save()


if __name__ == "__main__":
    main()
