import os
import json
import math
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd

# Prefer the first existing path among these locations
DATA_DIRS = ["datasets", "dataset", "."]


def _first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def resolve_path(name: str, synonyms: Optional[List[str]] = None) -> str:
    """Resolve a dataset file path robustly.

    Search order:
    - Each of DATA_DIRS joined with `name`
    - Each of DATA_DIRS joined with any provided `synonyms`
    - Bare `name` (CWD)
    - Bare synonyms (CWD)

    Additionally, if no `synonyms` provided and the name endswith `_test.csv`,
    try a fallback with that suffix removed (e.g., orders.csv).
    """
    names: List[str] = [name]
    if synonyms:
        names.extend(synonyms)
    else:
        # auto add simple fallback for *_test.csv naming
        if name.endswith("_test.csv"):
            names.append(name.replace("_test.csv", ".csv"))

    # search in preferred directories
    candidates: List[str] = []
    for base in names:
        for d in DATA_DIRS:
            candidates.append(os.path.join(d, base))
    # also allow bare filenames
    candidates.extend(names)

    resolved = _first_existing(candidates)
    return resolved if resolved is not None else os.path.join(DATA_DIRS[0], name)
import networkx as nx


def _read_csv_with_optional_dates(path: str, date_cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in date_cols:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c])
            except Exception:
                pass
    return df


def load_orders(path: str = "orders_test.csv") -> pd.DataFrame:
    rp = resolve_path(path, synonyms=["orders.csv"])
    df = _read_csv_with_optional_dates(
        rp,
        date_cols=[
            "created_at",
            "required_delivery",
            "earliest_pickup",
            "latest_pickup",
            "earliest_delivery",
            "latest_delivery",
        ],
    )
    return df


def load_nodes_edges(nodes_path: str = "nodes_test.csv", edges_path: str = "edges_test.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(resolve_path(nodes_path, synonyms=["nodes.csv"]))
    edges = pd.read_csv(resolve_path(edges_path, synonyms=["edges.csv"]))
    return nodes, edges


def load_weather(path: str = "weather.csv") -> Optional[pd.DataFrame]:
    rp = resolve_path(path)
    if not os.path.exists(rp):
        return None
    try:
        w = pd.read_csv(rp, parse_dates=["date"]) if os.path.getsize(rp) > 0 else None
    except Exception:
        # fallback if parse_dates fails
        w = pd.read_csv(rp)
        if w is not None and "date" in w.columns:
            try:
                w["date"] = pd.to_datetime(w["date"])
            except Exception:
                pass
    return w


def load_node_features(path: str = "node_features.csv") -> Optional[pd.DataFrame]:
    rp = resolve_path(path)
    if not os.path.exists(rp):
        return None
    try:
        df = pd.read_csv(rp)
        return df
    except Exception:
        return None


def ensure_daily_orders(df_orders: pd.DataFrame) -> pd.DataFrame:
    df = df_orders.copy()
    df["date"] = df["created_at"].dt.date
    daily = df.groupby("date").agg(
        n_orders=("order_id", "count"),
        mean_actual_time_h=("actual_time_h", "mean"),
        mean_actual_cost=("actual_cost", "mean"),
        mean_lateness_h=("lateness_h", "mean"),
        delayed_share=("status", lambda s: float((s == "delayed").mean())),
        cancelled_share=("status", lambda s: float((s == "cancelled").mean())),
    ).reset_index()
    return daily


def euclid_km(n1: pd.Series, n2: pd.Series) -> float:
    dx = float(n1["lon"]) - float(n2["lon"])
    dy = float(n1["lat"]) - float(n2["lat"])
    return float((dx * dx + dy * dy) ** 0.5 * 10.0)


def estimate_distance_matrix(nodes: pd.DataFrame) -> pd.DataFrame:
    ids = nodes["node_id"].tolist()
    rows = []
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if i == j:
                continue
            rows.append({
                "from_node": a,
                "to_node": b,
                "distance_km": euclid_km(nodes.loc[nodes.node_id == a].iloc[0], nodes.loc[nodes.node_id == b].iloc[0])
            })
    return pd.DataFrame(rows)


def build_graph_from_csvs(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, row in nodes.iterrows():
        if bool(row.get("is_active", True)):
            G.add_node(row["node_id"], lat=row["lat"], lon=row["lon"], type=row.get("type", None))
    for _, e in edges.iterrows():
        if not bool(e.get("is_active", True)):
            continue
        u = e["from_node"]; v = e["to_node"]
        if u not in G.nodes or v not in G.nodes:
            continue
        edge_id = e.get("edge_id", f"EDGE_{u}_{v}_{e.get('mode','road')}")
        G.add_edge(u, v,
                   edge_id=edge_id,
                   mode=e.get("mode", "road"),
                   distance_km=float(e.get("distance_km", 0.0)),
                   base_time_h=float(e.get("base_time_h", 0.0)),
                   base_cost=float(e.get("base_cost", 0.0)),
                   max_weight_kg=float(e.get("max_weight_kg", np.inf)),
                   max_volume_m3=float(e.get("max_volume_m3", np.inf)),
                   reliability=float(e.get("reliability", 0.95)))
    return G


def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d["dow"] = d[date_col].dt.weekday
    d["month"] = d[date_col].dt.month
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    return d


def join_weather_daily(daily: pd.DataFrame, weather: Optional[pd.DataFrame]) -> pd.DataFrame:
    if weather is None or len(weather) == 0:
        return daily
    # агрегируем погоду по дню (например, берем модальный тип и средние факторы)
    w = weather.copy()
    w["date"] = pd.to_datetime(w["date"]).dt.date
    # привести daily к тем же типам ключа
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    agg = w.groupby("date").agg(
        weather_mode=("weather", lambda s: s.mode().iat[0] if len(s.mode()) else "clear"),
        time_factor_mean=("time_factor", "mean"),
        cost_factor_mean=("cost_factor", "mean"),
    ).reset_index()
    out = daily.merge(agg, on="date", how="left")
    return out


def one_hot(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    return pd.get_dummies(df, columns=[col], prefix=prefix)

# --- Override join_weather_daily with a more robust implementation ---
def join_weather_daily(daily: pd.DataFrame, weather: Optional[pd.DataFrame]) -> pd.DataFrame:  # type: ignore[no-redef]
    if weather is None or len(weather) == 0:
        return daily
    w = weather.copy()
    try:
        w["date"] = pd.to_datetime(w["date"]).dt.date
    except Exception:
        pass
    out = daily.copy()
    try:
        out["date"] = pd.to_datetime(out["date"]).dt.date
    except Exception:
        pass
    # choose available columns
    label_col = "weather" if "weather" in w.columns else ("event" if "event" in w.columns else None)
    tf_col = "time_factor" if "time_factor" in w.columns else ("time_factor_road" if "time_factor_road" in w.columns else None)
    cf_col = "cost_factor" if "cost_factor" in w.columns else ("cost_factor_road" if "cost_factor_road" in w.columns else None)
    agg_spec = {}
    if label_col:
        agg_spec["weather_mode"] = (label_col, lambda s: s.mode().iat[0] if len(s.mode()) else "clear")
    if tf_col:
        agg_spec["time_factor_mean"] = (tf_col, "mean")
    if cf_col:
        agg_spec["cost_factor_mean"] = (cf_col, "mean")
    if not agg_spec:
        return out
    agg = w.groupby("date").agg(**agg_spec).reset_index()
    out = out.merge(agg, on="date", how="left")
    return out
