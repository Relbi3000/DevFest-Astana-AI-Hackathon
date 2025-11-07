"""
Graph Neural Network models for edge time/cost prediction.

Implements a simple GraphSAGE-style message passing network (pure PyTorch) that
learns node embeddings from the transportation graph and predicts per-edge time
and cost using an MLP over [h_u, h_v, edge_features].

Artifacts:
- models/edge_gnn.pt (state dict + config)
"""
import os
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore

try:
    from .utils import load_nodes_edges, build_graph_from_csvs, resolve_path
except ImportError:
    from utils import load_nodes_edges, build_graph_from_csvs, resolve_path


def _build_order_segments_from_orders(orders: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    idx: dict[tuple[frozenset[str], str], dict] = {}
    for _, e in edges.iterrows():
        key = (frozenset([str(e["from_node"]), str(e["to_node"]) ]), str(e.get("mode", "road")))
        idx[key] = e.to_dict()
    rows: list[dict] = []
    def _split(s: str) -> list[str]:
        if pd.isna(s) or s is None:
            return []
        return [x for x in str(s).split(";") if len(x)]
    for _, r in orders.iterrows():
        oid = r.get("order_id")
        nodes = _split(r.get("route_nodes"))
        modes = _split(r.get("route_modes"))
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
                "max_weight_kg": float(ed.get("max_weight_kg", np.inf)) if ed else np.inf,
                "max_volume_m3": float(ed.get("max_volume_m3", np.inf)) if ed else np.inf,
                "reliability": float(ed.get("reliability", 0.95)) if ed else 0.95,
            }
            rows.append(row)
    seg = pd.DataFrame(rows)
    return seg


def _load_required() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    import pandas as _pd
    orders = _pd.read_csv(resolve_path("orders_test.csv", synonyms=["orders.csv"]))
    if "created_at" in orders.columns:
        try:
            orders["created_at"] = _pd.to_datetime(orders["created_at"]).dt.tz_localize(None)
        except Exception:
            pass
    seg_path = resolve_path("order_segments.csv")
    try:
        seg = _pd.read_csv(seg_path)
    except Exception:
        seg = None
    nodes, edges = load_nodes_edges()
    if seg is None or len(seg) == 0:
        seg = _build_order_segments_from_orders(orders, edges)
    # weather optional
    weather = None
    wp = resolve_path("weather.csv")
    if os.path.exists(wp):
        try:
            weather = _pd.read_csv(wp, parse_dates=["date"]) if os.path.getsize(wp) > 0 else None
        except Exception:
            w = _pd.read_csv(wp)
            if w is not None and "date" in w.columns:
                try:
                    w["date"] = _pd.to_datetime(w["date"]) 
                except Exception:
                    pass
            weather = w
    # normalize weather factors
    if weather is not None and len(weather) > 0:
        if "time_factor" not in weather.columns and "time_factor_road" in weather.columns:
            weather = weather.copy(); weather["time_factor"] = weather["time_factor_road"]
        if "cost_factor" not in weather.columns and "cost_factor_road" in weather.columns:
            weather = weather.copy(); weather["cost_factor"] = weather["cost_factor_road"]
    return seg, orders, edges, weather


def _prepare_training() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], pd.Series, Dict[str, int]]:
    seg, orders, edges, weather = _load_required()
    # Aggregate base sums per order to split targets
    seg_sum = seg.groupby("order_id").agg(sum_base_time=("base_time_h", "sum"), sum_base_cost=("base_cost", "sum")).reset_index()
    df = seg.merge(seg_sum, on="order_id", how="left")
    # Join order-level attributes
    keep = [
        "order_id", "origin_id", "destination_id", "created_at",
        "weight_kg", "volume_m3", "cargo_class", "actual_time_h", "actual_cost",
    ]
    df = df.merge(orders[keep], on="order_id", how="left")
    # Weather per segment start node
    if weather is not None and len(weather) > 0:
        w = weather.copy(); w["date"] = pd.to_datetime(w["date"]).dt.date
        df["date"] = pd.to_datetime(df["created_at"]).dt.date
        df = df.merge(
            w[["date", "node_id", "time_factor", "cost_factor"]],
            left_on=["date", "from_node"], right_on=["date", "node_id"], how="left",
        )
        if "node_id" in df.columns:
            df = df.drop(columns=["node_id"]) 
    else:
        df["time_factor"] = 1.0; df["cost_factor"] = 1.0
    # Targets (proportional split)
    eps = 1e-6
    df["time_target"] = df["actual_time_h"] * (df["base_time_h"] / (df["sum_base_time"] + eps))
    df["cost_target"] = df["actual_cost"] * (df["base_cost"] / (df["sum_base_cost"] + eps))
    # Edge feature columns (keep minimal, numeric)
    feat_cols = [
        "distance_km", "base_time_h", "base_cost", "max_weight_kg", "max_volume_m3", "reliability",
        "time_factor", "cost_factor", "weight_kg", "volume_m3",
        "mode_road", "mode_air", "mode_combined",
    ]
    # one-hot mode
    df["mode_road"] = (df["mode"].astype(str) == "road").astype(float)
    df["mode_air"] = (df["mode"].astype(str) == "air").astype(float)
    df["mode_combined"] = (df["mode"].astype(str) == "combined").astype(float)
    X = df[feat_cols].fillna(0.0).reset_index(drop=True)
    y_time = df["time_target"].fillna(0.0).values
    y_cost = df["cost_target"].fillna(0.0).values
    created_at = pd.to_datetime(df["created_at"]) if "created_at" in df.columns else pd.Series([pd.Timestamp(0)] * len(df))
    # node index mapping
    nodes, edges_df = load_nodes_edges()
    node_ids = nodes["node_id"].astype(str).tolist()
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    # Attach indices
    df["u_idx"] = df["from_node"].astype(str).map(node_index)
    df["v_idx"] = df["to_node"].astype(str).map(node_index)
    idx_uv = df[["u_idx", "v_idx"]].fillna(-1).astype(int)
    X["u_idx"] = idx_uv["u_idx"]
    X["v_idx"] = idx_uv["v_idx"]
    return X, y_time, y_cost, feat_cols, created_at, node_index


class GraphSAGE(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int = 32):
        super().__init__()
        self.emb0 = nn.Embedding(num_nodes, emb_dim)
        self.W1 = nn.Linear(emb_dim, emb_dim)
        self.W2 = nn.Linear(emb_dim, emb_dim)
        self.act = nn.ReLU()

    def forward(self, adj: List[List[int]]) -> torch.Tensor:
        # Compute 2-layer mean aggregation embeddings
        N = len(adj)
        H0 = self.emb0.weight
        # layer 1
        H1 = torch.zeros_like(H0)
        for i in range(N):
            neigh = adj[i]
            if neigh:
                m = H0[neigh].mean(dim=0)
                H1[i] = self.act(self.W1((H0[i] + m) / 2.0))
            else:
                H1[i] = self.act(self.W1(H0[i]))
        # layer 2
        H2 = torch.zeros_like(H1)
        for i in range(N):
            neigh = adj[i]
            if neigh:
                m = H1[neigh].mean(dim=0)
                H2[i] = self.act(self.W2((H1[i] + m) / 2.0))
            else:
                H2[i] = self.act(self.W2(H1[i]))
        return H2


class EdgePredictor(nn.Module):
    def __init__(self, node_emb_dim: int, edge_feat_dim: int):
        super().__init__()
        in_dim = node_emb_dim * 2 + edge_feat_dim
        hidden = max(64, node_emb_dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2)  # [time, cost]
        )
    def forward(self, hu: torch.Tensor, hv: torch.Tensor, xe: torch.Tensor) -> torch.Tensor:
        z = torch.cat([hu, hv, xe], dim=-1)
        return self.mlp(z)


def _build_adj_list() -> Tuple[List[List[int]], Dict[str, int]]:
    nodes, edges = load_nodes_edges()
    node_ids = nodes["node_id"].astype(str).tolist()
    idx = {nid: i for i, nid in enumerate(node_ids)}
    adj = [[] for _ in node_ids]
    for _, e in edges.iterrows():
        u = str(e["from_node"]); v = str(e["to_node"])
        if u in idx and v in idx:
            ui = idx[u]; vi = idx[v]
            if vi not in adj[ui]:
                adj[ui].append(vi)
            if ui not in adj[vi]:
                adj[vi].append(ui)
    return adj, idx


def train_edge_gnn(epochs: int = 30, emb_dim: int = 32) -> Optional[Dict[str, object]]:
    if torch is None:
        return None
    X, y_time, y_cost, feat_cols, created_at, node_index = _prepare_training()
    # temporal split
    try:
        order_idx = created_at.sort_values().index
        X = X.loc[order_idx].reset_index(drop=True)
        y_time = y_time[order_idx]
        y_cost = y_cost[order_idx]
    except Exception:
        pass
    n = len(X)
    if n < 50:
        return None
    split = int(n * 0.8)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr_t, yte_t = y_time[:split], y_time[split:]
    ytr_c, yte_c = y_cost[:split], y_cost[split:]

    # Build adjacency list from current graph
    adj, idx_map = _build_adj_list()
    N = len(adj)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sage = GraphSAGE(num_nodes=N, emb_dim=emb_dim).to(device)
    pred = EdgePredictor(node_emb_dim=emb_dim, edge_feat_dim=len(feat_cols)).to(device)
    params = list(sage.parameters()) + list(pred.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    loss_fn = nn.L1Loss()

    # tensors
    Xtr_edge = torch.tensor(Xtr[feat_cols].values, dtype=torch.float32, device=device)
    Utr = torch.tensor(Xtr["u_idx"].values, dtype=torch.long, device=device)
    Vtr = torch.tensor(Xtr["v_idx"].values, dtype=torch.long, device=device)
    ytr = torch.stack([
        torch.tensor(ytr_t, dtype=torch.float32, device=device),
        torch.tensor(ytr_c, dtype=torch.float32, device=device)
    ], dim=1)
    Xte_edge = torch.tensor(Xte[feat_cols].values, dtype=torch.float32, device=device)
    Ute = torch.tensor(Xte["u_idx"].values, dtype=torch.long, device=device)
    Vte = torch.tensor(Xte["v_idx"].values, dtype=torch.long, device=device)
    yte = torch.stack([
        torch.tensor(yte_t, dtype=torch.float32, device=device),
        torch.tensor(yte_c, dtype=torch.float32, device=device)
    ], dim=1)

    # convert adj to tensors of neighbors indices per node
    adj_t = [torch.tensor(nei if nei else [], dtype=torch.long, device=device) for nei in adj]

    def compute_embeddings() -> torch.Tensor:
        # materialize neighbor lists on CPU indices for loops
        adj_lists: List[List[int]] = [list(map(int, a.tolist())) if isinstance(a, torch.Tensor) else list(a) for a in adj_t]
        return sage(adj_lists)

    best_val = float('inf')
    for _ in range(epochs):
        sage.train(); pred.train()
        H = compute_embeddings()
        hu = H[Utr]; hv = H[Vtr]
        yh = pred(hu, hv, Xtr_edge)
        loss = loss_fn(yh, ytr)
        opt.zero_grad(); loss.backward(); opt.step()
        # val
        sage.eval(); pred.eval()
        with torch.no_grad():
            Ht = compute_embeddings()
            hvu = Ht[Ute]; hvv = Ht[Vte]
            yhv = pred(hvu, hvv, Xte_edge)
            val = float(loss_fn(yhv, yte).item())
            if val < best_val:
                best_val = val
                best = {
                    "sage": {k: v.detach().cpu() for k, v in sage.state_dict().items()},
                    "pred": {k: v.detach().cpu() for k, v in pred.state_dict().items()},
                }
    if best_val == float('inf'):
        return None
    os.makedirs("models", exist_ok=True)
    payload = {
        "sage_state": best["sage"],
        "pred_state": best["pred"],
        "emb_dim": int(emb_dim),
        "feat_cols": list(feat_cols),
        "node_index": node_index,
    }
    torch.save(payload, "models/edge_gnn.pt")
    with open("models/edge_gnn_config.json", "w", encoding="utf-8") as f:
        json.dump({"emb_dim": int(emb_dim), "feat_cols": list(feat_cols)}, f, ensure_ascii=False, indent=2)
    # compute final val metrics (time/cost MAE/MAPE)
    sage.eval(); pred.eval()
    with torch.no_grad():
        Ht = sage(adj)
        yhv = pred(Ht[Ute], Ht[Vte], Xte_edge)
    yt_pred = yhv[:, 0].cpu().numpy(); yc_pred = yhv[:, 1].cpu().numpy()
    yt_true = yte[:, 0].cpu().numpy(); yc_true = yte[:, 1].cpu().numpy()
    eps = 1e-6
    mae_t = float(np.mean(np.abs(yt_true - yt_pred)))
    mae_c = float(np.mean(np.abs(yc_true - yc_pred)))
    # robust MAPE: игнорируем почти нулевые таргеты
    thr_t = max(1e-3, 0.01 * float(np.median(np.abs(yt_true)) if len(yt_true) else 0.0))
    thr_c = max(1e-3, 0.01 * float(np.median(np.abs(yc_true)) if len(yc_true) else 0.0))
    m_t = np.abs(yt_true) >= thr_t
    m_c = np.abs(yc_true) >= thr_c
    if np.any(m_t):
        mape_t = float(np.mean(np.abs((yt_true[m_t] - yt_pred[m_t]) / np.clip(np.abs(yt_true[m_t]), eps, None))) * 100.0)
    else:
        mape_t = float('nan')
    if np.any(m_c):
        mape_c = float(np.mean(np.abs((yc_true[m_c] - yc_pred[m_c]) / np.clip(np.abs(yc_true[m_c]), eps, None))) * 100.0)
    else:
        mape_c = float('nan')
    metrics = {"val_l1": float(best_val), "mae_time": mae_t, "mae_cost": mae_c, "mape_time": mape_t, "mape_cost": mape_c, "n_train": int(split), "n_test": int(n - split)}
    with open("models/edge_gnn_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


class EdgeGNNInferencer:
    def __init__(self, sage: GraphSAGE, pred: EdgePredictor, feat_cols: List[str], node_index: Dict[str, int]):
        self.sage = sage
        self.pred = pred
        self.feat_cols = feat_cols
        self.node_index = node_index
        self._H = None
        self._adj: Optional[List[List[int]]] = None

    @staticmethod
    def load(model_path: str = "models/edge_gnn.pt") -> Optional["EdgeGNNInferencer"]:
        if torch is None:
            return None
        if not os.path.exists(model_path):
            return None
        payload = torch.load(model_path, map_location="cpu")
        emb_dim = int(payload["emb_dim"])
        feat_cols = list(payload["feat_cols"]) 
        node_index = dict(payload["node_index"]) if "node_index" in payload else {}
        # Build models
        adj, idx_map = _build_adj_list()
        N = len(adj)
        sage = GraphSAGE(num_nodes=N, emb_dim=emb_dim)
        pred = EdgePredictor(node_emb_dim=emb_dim, edge_feat_dim=len(feat_cols))
        try:
            sage.load_state_dict(payload["sage_state"], strict=False)
            pred.load_state_dict(payload["pred_state"], strict=False)
        except Exception:
            return None
        inf = EdgeGNNInferencer(sage, pred, feat_cols, node_index)
        inf._adj = adj
        return inf

    def _ensure_embeddings(self):
        if self._H is None:
            self.sage.eval()
            self._H = self.sage(self._adj or [])  # type: ignore[arg-type]

    def predict(self, u: str, v: str, edge_attrs: Dict[str, float]) -> Tuple[float, float]:
        self._ensure_embeddings()
        ui = self.node_index.get(str(u))
        vi = self.node_index.get(str(v))
        if ui is None or vi is None or self._H is None:
            return float(edge_attrs.get("base_time_h", 0.0)), float(edge_attrs.get("base_cost", 0.0))
        # build edge feature vector
        x = np.array([edge_attrs.get(c, 0.0) for c in self.feat_cols], dtype=np.float32)
        xe = torch.tensor(x).unsqueeze(0)
        hu = self._H[ui].unsqueeze(0)
        hv = self._H[vi].unsqueeze(0)
        self.pred.eval()
        with torch.no_grad():
            y = self.pred(hu, hv, xe)
        yt = float(y[0, 0].item()); yc = float(y[0, 1].item())
        return yt, yc


def main():
    res = train_edge_gnn()
    if res is None:
        print("GNN training skipped (torch unavailable or insufficient data).")
    else:
        print("Edge GNN trained:", res)


if __name__ == "__main__":
    main()
