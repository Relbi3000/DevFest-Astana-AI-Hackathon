"""
Route optimizer baseline.

Implements:
- Building/reading a graph (prefers graph.gpickle if present).
- Capacity-aware filtering.
- k-shortest simple paths via time-weighted cost.
- Multicriteria scoring: score = w1*(1/time) + w2*(1/cost) + w3*reliability.

Provides optimize_delivery(...) returning candidate routes with segments and scores.
"""
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx

# Support both `python -m ml.route_optimizer` and direct `python ml/route_optimizer.py`
try:
    from .utils import load_nodes_edges, build_graph_from_csvs
    from .utils import load_weather, resolve_path
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import load_nodes_edges, build_graph_from_csvs
    from utils import load_weather, resolve_path

# Optional GNN inferencer for edge predictions
try:
    from .gnn_models import EdgeGNNInferencer
except Exception:
    EdgeGNNInferencer = None  # type: ignore


def _load_graph() -> nx.Graph:
    gp = resolve_path("graph.gpickle")
    if os.path.exists(gp):
        try:
            return nx.read_gpickle(gp)
        except Exception:
            pass
    nodes, edges = load_nodes_edges()
    return build_graph_from_csvs(nodes, edges)


def _capacity_filter(G: nx.Graph, weight_kg: Optional[float], volume_m3: Optional[float]) -> nx.Graph:
    if weight_kg is None and volume_m3 is None:
        return G
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        ok = True
        if weight_kg is not None and float(d.get("max_weight_kg", np.inf)) < float(weight_kg):
            ok = False
        if volume_m3 is not None and float(d.get("max_volume_m3", np.inf)) < float(volume_m3):
            ok = False
        if ok:
            H.add_edge(u, v, **d)
    return H


def _time_weight(u, v, d) -> float:
    return float(d.get("base_time_h", 1.0))


def _load_edge_models():
    time_model = None; cost_model = None; feat_cols = None
    time_q = None; cost_q = None
    gnn_inf = None
    try:
        import pickle
        with open("models/edge_time_model.pkl", "rb") as f:
            t = pickle.load(f); time_model = t.get("model"); feat_cols = t.get("feature_cols")
        with open("models/edge_cost_model.pkl", "rb") as f:
            c = pickle.load(f); cost_model = c.get("model")
        # optional quantile models
        try:
            with open("models/edge_time_model_quantiles.pkl", "rb") as f:
                tq = pickle.load(f); time_q = tq.get("models")
        except Exception:
            time_q = None
        try:
            with open("models/edge_cost_model_quantiles.pkl", "rb") as f:
                cq = pickle.load(f); cost_q = cq.get("models")
        except Exception:
            cost_q = None
    except Exception:
        pass
    # Try GNN inferencer
    try:
        if EdgeGNNInferencer is not None:
            gnn_inf = EdgeGNNInferencer.load("models/edge_gnn.pt")
    except Exception:
        gnn_inf = None
    return time_model, cost_model, feat_cols, time_q, cost_q, gnn_inf


def _segment_features(u: str, v: str, d: Dict, weight_kg: Optional[float], volume_m3: Optional[float], time_factor: float, cost_factor: float) -> Dict:
    row = {
        "distance_km": float(d.get("distance_km", 0.0)),
        "base_time_h": float(d.get("base_time_h", 0.0)),
        "base_cost": float(d.get("base_cost", 0.0)),
        "max_weight_kg": float(d.get("max_weight_kg", np.inf)),
        "max_volume_m3": float(d.get("max_volume_m3", np.inf)),
        "reliability": float(d.get("reliability", 0.95)),
        "time_factor": float(time_factor),
        "cost_factor": float(cost_factor),
        "weight_kg": float(weight_kg or 0.0),
        "volume_m3": float(volume_m3 or 0.0),
        # mode one-hot minimal: we will add two keys
        f"mode_{d.get('mode','road')}": 1.0,
        "mode_road": 1.0 if d.get("mode","road") == "road" else 0.0,
        "mode_air": 1.0 if d.get("mode","road") == "air" else 0.0,
        "mode_combined": 1.0 if d.get("mode","road") == "combined" else 0.0,
    }
    return row


def _path_stats(G: nx.Graph, path: List[str], mode_pref: str = "balanced", *, weight_kg: Optional[float] = None, volume_m3: Optional[float] = None, created_at: Optional[pd.Timestamp] = None, weather_df: Optional[pd.DataFrame] = None, scenarios: int = 1) -> Dict:
    time_h = 0.0
    cost = 0.0
    reliability = 1.0
    modes: List[str] = []
    edge_ids: List[str] = []
    segments: List[Dict] = []
    transfer_time_h = 2.0
    transfer_cost = 8000.0
    prev_mode = None
    # load predictors if present
    time_model, cost_model, feat_cols, time_q, cost_q, gnn_inf = _load_edge_models()
    have_q = isinstance(time_q, dict) and isinstance(cost_q, dict)
    # weather factors per segment (use created_at date + segment from_node if available)
    date_key = None
    if created_at is not None:
        try:
            date_key = pd.to_datetime(created_at).date()
        except Exception:
            date_key = None
    # scenario samples accumulators (optional)
    time_samples = np.zeros(scenarios, dtype=float) if scenarios and scenarios > 1 else None
    cost_samples = np.zeros(scenarios, dtype=float) if scenarios and scenarios > 1 else None
    rng = np.random.default_rng(42)

    for i, (u, v) in enumerate(zip(path, path[1:])):
        d = G[u][v]
        base_t = float(d.get("base_time_h", 0.0))
        base_c = float(d.get("base_cost", 0.0))
        dist_km = float(d.get("distance_km", 0.0))
        current_mode = d.get("mode", "road")
        eid = d.get("edge_id", f"EDGE_{u}_{v}_{current_mode}")
        # default weather factors
        t_fac = 1.0; c_fac = 1.0
        if weather_df is not None and date_key is not None:
            try:
                roww = weather_df[(weather_df["date"].dt.date == date_key) & (weather_df["node_id"] == u)]
                if len(roww) > 0:
                    # support alternative factor columns
                    r0 = roww.iloc[0]
                    t_fac = float(r0["time_factor"]) if "time_factor" in r0 else float(r0["time_factor_road"]) if "time_factor_road" in r0 else 1.0
                    c_fac = float(r0["cost_factor"]) if "cost_factor" in r0 else float(r0["cost_factor_road"]) if "cost_factor_road" in r0 else 1.0
            except Exception:
                pass
        # predictions preference: GNN > boosted trees > base
        feats = _segment_features(u, v, d, weight_kg, volume_m3, t_fac, c_fac)
        if gnn_inf is not None:
            try:
                pred_t, pred_c = gnn_inf.predict(u, v, feats)
            except Exception:
                pred_t, pred_c = base_t * t_fac, base_c * c_fac
        elif time_model is not None and cost_model is not None and feat_cols is not None:
            import pandas as _pd
            row = {col: feats.get(col, 0.0) for col in feat_cols}
            xdf = _pd.DataFrame([row], columns=feat_cols)
            try:
                pred_t = float(time_model.predict(xdf)[0])
                pred_c = float(cost_model.predict(xdf)[0])
            except Exception:
                pred_t, pred_c = base_t * t_fac, base_c * c_fac
        else:
            pred_t, pred_c = base_t * t_fac, base_c * c_fac
        time_h += pred_t
        cost += pred_c
        # scenario sampling: derive per-segment uncertainty from quantiles when available
        if time_samples is not None and cost_samples is not None:
            if have_q and feat_cols is not None:
                try:
                    import pandas as _pd
                    row_q = {col: feats.get(col, 0.0) for col in feat_cols}
                    xq = _pd.DataFrame([row_q], columns=feat_cols)
                    t_q10 = float(time_q.get("0.1").predict(xq)[0]) if time_q.get("0.1") else pred_t
                    t_q90 = float(time_q.get("0.9").predict(xq)[0]) if time_q.get("0.9") else pred_t
                    c_q10 = float(cost_q.get("0.1").predict(xq)[0]) if cost_q.get("0.1") else pred_c
                    c_q90 = float(cost_q.get("0.9").predict(xq)[0]) if cost_q.get("0.9") else pred_c
                    # approximate variance from IQR under normal assumption
                    t_sigma = max((t_q90 - t_q10) / 2.56, 1e-6)
                    c_sigma = max((c_q90 - c_q10) / 2.56, 1e-6)
                except Exception:
                    t_sigma = max(0.15 * pred_t, 1e-6); c_sigma = max(0.2 * pred_c, 1e-6)
            else:
                t_sigma = max(0.15 * pred_t, 1e-6)
                c_sigma = max(0.2 * pred_c, 1e-6)
            # sample normal noise (non-negative by clipping)
            time_samples += np.clip(rng.normal(loc=pred_t, scale=t_sigma, size=scenarios), 0.0, None)
            cost_samples += np.clip(rng.normal(loc=pred_c, scale=c_sigma, size=scenarios), 0.0, None)
        reliability *= float(d.get("reliability", 0.95))
        if mode_pref == "combined" and prev_mode is not None and current_mode != prev_mode:
            time_h += transfer_time_h
            cost += transfer_cost
        prev_mode = current_mode
        modes.append(current_mode)
        edge_ids.append(str(eid))
        segments.append({
            "seq": i,
            "from_node": u,
            "to_node": v,
            "mode": current_mode,
            "edge_id": eid,
            "distance_km": dist_km,
            "base_time_h": base_t,
            "base_cost": base_c,
            "pred_time_h": pred_t,
            "pred_cost": pred_c,
            "reliability": float(d.get("reliability", 0.95)),
        })
    out = {
        "time_h": float(time_h),
        "cost": float(cost),
        "reliability": float(reliability),
        "nodes": path,
        "modes": modes,
        "edge_ids": edge_ids,
        "segments": segments,
        "n_segments": len(path) - 1,
    }
    if time_samples is not None and cost_samples is not None:
        # aggregate stochastic summaries
        t_sorted = np.sort(time_samples)
        c_sorted = np.sort(cost_samples)
        p90_idx = int(0.9 * (scenarios - 1))
        p95_idx = int(0.95 * (scenarios - 1))
        out.update({
            "time_mean": float(np.mean(time_samples)),
            "time_p90": float(t_sorted[p90_idx]),
            "time_cvar95": float(np.mean(t_sorted[p95_idx:])),
            "cost_mean": float(np.mean(cost_samples)),
            "cost_p90": float(c_sorted[p90_idx]),
            "cost_cvar95": float(np.mean(c_sorted[p95_idx:])),
        })
    return out


def _multicriteria_score(route: Dict, w_time: float = 1.0, w_cost: float = 1.0, w_rel: float = 1.0, lambda_cvar: float = 0.0) -> float:
    # prefer expected metrics if available
    t = route.get("time_mean", route.get("time_h", 1.0))
    c = route.get("cost_mean", route.get("cost", 1.0))
    r = max(min(route.get("reliability", 1.0), 1.0), 1e-6)
    score = float(w_time * (1.0 / max(t, 1e-6)) + w_cost * (1.0 / max(c, 1e-6)) + w_rel * r)
    # risk aversion via CVaR penalty if provided
    if lambda_cvar > 0.0:
        t_cvar = route.get("time_cvar95", t)
        c_cvar = route.get("cost_cvar95", c)
        score -= float(lambda_cvar * ((t_cvar / max(t, 1e-6)) + (c_cvar / max(c, 1e-6))) / 2.0)
    return score


def _softmax(x):
    import numpy as _np
    x = _np.array(x, dtype=float)
    x = x - _np.max(x)
    ex = _np.exp(x)
    s = _np.sum(ex)
    return ex / (s if s > 0 else 1.0)


def _quantum_simulated_paths(
    G: nx.Graph,
    origin: str,
    destination: str,
    *,
    beam_width: int = 4,
    sweeps: int = 50,
    temp_start: float = 1.5,
    temp_end: float = 0.2,
    max_depth: int = 32,
) -> list[list[str]]:
    """Quantum-inspired stochastic beam search (simulation).

    Keeps a beam of partial paths; at each step expands by sampling neighbor moves
    with softmax over a heuristic (fast time weight), annealing temperature schedule.
    Returns a list of complete paths to destination (best-effort, may be empty).
    """
    import numpy as _np
    if origin not in G or destination not in G:
        return []
    rng = _np.random.default_rng(1337)
    def heuristic_time(path: list[str]) -> float:
        if len(path) <= 1:
            return 0.0
        t = 0.0
        for u, v in zip(path, path[1:]):
            t += float(G[u][v].get("base_time_h", 1.0))
        return t
    best_paths: list[list[str]] = []
    for s in range(sweeps):
        T = temp_start + (temp_end - temp_start) * (s / max(1, sweeps - 1))
        beam: list[list[str]] = [[origin]]
        for _ in range(max_depth):
            # collect expansions
            cand: list[list[str]] = []
            for p in beam:
                u = p[-1]
                if u == destination:
                    cand.append(p)
                    continue
                neigh = list(G.neighbors(u))
                if not neigh:
                    continue
                # scores: favor small base_time, discourage revisits
                raw = []
                for v in neigh:
                    if v in p:  # avoid cycles
                        raw.append(-1e3)
                    else:
                        raw.append(-float(G[u][v].get("base_time_h", 1.0)))
                prob = _softmax(_np.array(raw) / max(T, 1e-6))
                # sample up to 2 candidates per beam path
                k = min(2, len(neigh))
                picks = rng.choice(len(neigh), size=k, replace=False, p=prob)
                for idx in picks:
                    v = neigh[int(idx)]
                    cand.append(p + [v])
            if not cand:
                break
            # sort by heuristic and keep top beam_width
            cand.sort(key=heuristic_time)
            beam = cand[:beam_width]
            # collect any finished
            done = [p for p in beam if p[-1] == destination]
            best_paths.extend(done)
            if done:
                break
    # de-duplicate
    uniq = []
    seen = set()
    for p in best_paths:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq[:beam_width]


def _reliability_greedy_path(G: nx.Graph, origin: str, destination: str, max_steps: int = 200) -> list[str]:
    if origin not in G or destination not in G:
        return []
    cur = origin
    path = [cur]
    visited = {cur}
    steps = 0
    while cur != destination and steps < max_steps:
        steps += 1
        neigh = list(G.neighbors(cur))
        if not neigh:
            break
        best = None
        best_rel = -1.0
        for nb in neigh:
            if nb in visited:
                continue
            rel = float(G[cur][nb].get("reliability", 0.0))
            if rel > best_rel:
                best_rel = rel
                best = nb
        if best is None:
            break
        cur = best
        path.append(cur)
        visited.add(cur)
    return path if path and path[-1] == destination else []


def optimize_delivery(
    origin: str,
    destination: str,
    cargo_specs: Dict,
    constraints: Dict,
    k_paths: int = 3,
) -> List[Dict]:
    """
    Returns ranked candidate routes.
    cargo_specs: {weight_kg, volume_m3, mode_pref}
    constraints: {w_time, w_cost, w_rel}
    """
    G = _load_graph()
    weight_kg = cargo_specs.get("weight_kg")
    volume_m3 = cargo_specs.get("volume_m3")
    mode_pref = cargo_specs.get("mode_pref", "balanced")
    created_at = cargo_specs.get("created_at")
    weather_df = load_weather()

    Gc = _capacity_filter(G, weight_kg, volume_m3)
    # RL route (influential by default)
    use_rl = constraints.get("use_rl")
    if use_rl is None:
        use_rl = True
    if use_rl and rl_plan_route is not None:
        try:
            rl_path = rl_plan_route(origin, destination, max_steps=constraints.get("rl_max_steps", 200))
        except Exception:
            rl_path = []
        if rl_path:
            stats = _path_stats(Gc, rl_path, mode_pref=mode_pref, weight_kg=weight_kg, volume_m3=volume_m3, created_at=created_at, weather_df=weather_df, scenarios=scenarios)
            # constraints checks with small relaxation margin for RL path
            ok = True
            relax = float(constraints.get("rl_relax", 0.0))  # e.g., 0.05 = 5% relaxation
            if constraints.get("max_transfers") is not None:
                transfers = sum(1 for i in range(1, len(stats["modes"])) if stats["modes"][i] != stats["modes"][i-1])
                if transfers > int(constraints["max_transfers"]):
                    ok = False
            if constraints.get("deadline_hours") is not None:
                tchk = stats.get("time_p90", stats.get("time_h", 0.0))
                if tchk > float(constraints["deadline_hours"]) * (1.0 + relax):
                    ok = False
            if ok:
                stats["score"] = _multicriteria_score(stats, w_time=constraints.get("w_time", 1.0), w_cost=constraints.get("w_cost", 1.0), w_rel=constraints.get("w_rel", 1.0), lambda_cvar=float(constraints.get("lambda_cvar", 0.0)))
                return [stats]
            else:
                # keep RL candidate with penalty; will be compared with others
                penalty = 0.1  # subtract 10% from score to reflect violation risk
                stats["score"] = _multicriteria_score(stats, w_time=constraints.get("w_time", 1.0), w_cost=constraints.get("w_cost", 1.0), w_rel=constraints.get("w_rel", 1.0), lambda_cvar=float(constraints.get("lambda_cvar", 0.0))) * (1.0 - penalty)
                rl_candidate = stats
        else:
            rl_candidate = None
    else:
        rl_candidate = None
    # pick weighted graph (time-based) fallback
    try:
        gen_paths = nx.shortest_simple_paths(Gc, origin, destination, weight=_time_weight)
    except nx.NetworkXNoPath:
        return []

    # constraints
    deadline_h = constraints.get("deadline_hours")
    max_transfers = constraints.get("max_transfers")
    forbidden_nodes = set(constraints.get("forbidden_nodes", []) or [])
    forbidden_edges = set(constraints.get("forbidden_edges", []) or [])
    scenarios = int(constraints.get("scenarios", 1))

    def path_is_forbidden(p: List[str]) -> bool:
        if forbidden_nodes:
            if any(n in forbidden_nodes for n in p):
                return True
        if forbidden_edges:
            for u, v in zip(p, p[1:]):
                eid = Gc[u][v].get("edge_id")
                if str(eid) in forbidden_edges:
                    return True
        return False

    candidates: List[Dict] = []
    for idx, path in enumerate(gen_paths):
        if path_is_forbidden(path):
            continue
        stats = _path_stats(Gc, path, mode_pref=mode_pref, weight_kg=weight_kg, volume_m3=volume_m3, created_at=created_at, weather_df=weather_df, scenarios=scenarios)
        # check transfers constraint
        if max_transfers is not None:
            transfers = sum(1 for i in range(1, len(stats["modes"])) if stats["modes"][i] != stats["modes"][i-1])
            if transfers > int(max_transfers):
                continue
        # check deadline using P90 if stochastic samples exist
        if deadline_h is not None:
            time_check = stats.get("time_p90", stats.get("time_h", 0.0))
            if time_check > float(deadline_h):
                continue
        candidates.append(stats)
        if len(candidates) >= k_paths:
            break

    # Advanced: quantum-inspired candidates
    if constraints.get("use_quantum", False):
        q_beam = int(constraints.get("quant_beam", 4))
        q_sweeps = int(constraints.get("quant_sweeps", 30))
        q_paths = _quantum_simulated_paths(Gc, origin, destination, beam_width=q_beam, sweeps=q_sweeps)
        for p in q_paths:
            if path_is_forbidden(p):
                continue
            stats = _path_stats(Gc, p, mode_pref=mode_pref, weight_kg=weight_kg, volume_m3=volume_m3, created_at=created_at, weather_df=weather_df, scenarios=scenarios)
            if deadline_h is not None:
                time_check = stats.get("time_p90", stats.get("time_h", 0.0))
                if time_check > float(deadline_h):
                    continue
            candidates.append(stats)

    # Advanced: multi-agent (reliability-greedy + baseline)
    if constraints.get("use_multi_agent", False):
        rel_path = _reliability_greedy_path(Gc, origin, destination)
        if rel_path:
            stats = _path_stats(Gc, rel_path, mode_pref=mode_pref, weight_kg=weight_kg, volume_m3=volume_m3, created_at=created_at, weather_df=weather_df, scenarios=scenarios)
            candidates.append(stats)

    w_time = constraints.get("w_time", 1.0)
    w_cost = constraints.get("w_cost", 1.0)
    w_rel = constraints.get("w_rel", 1.0)
    lambda_cvar = float(constraints.get("lambda_cvar", 0.0))

    # Optional RL-style tuning of weights via epsilon-greedy over a small discrete set
    if constraints.get("rl_tune", False) and len(candidates) > 1:
        weight_arms = [
            (1.0, 1.0, 1.0), (1.5, 1.0, 0.8), (0.8, 1.5, 1.0), (1.2, 0.8, 1.5),
            (2.0, 1.0, 0.5), (1.0, 2.0, 0.5), (0.5, 1.0, 2.0)
        ]
        q_values = {w: 0.0 for w in weight_arms}
        counts = {w: 0 for w in weight_arms}
        rng = np.random.default_rng(123)
        episodes = int(constraints.get("rl_episodes", 50))
        eps = float(constraints.get("rl_epsilon", 0.2))
        for _ in range(episodes):
            if rng.random() < eps:
                w = weight_arms[int(rng.integers(0, len(weight_arms)))]
            else:
                w = max(q_values.keys(), key=lambda k: q_values[k])
            wt, wc, wr = w
            # reward: best route score under these weights
            reward = max(_multicriteria_score(c, w_time=wt, w_cost=wc, w_rel=wr, lambda_cvar=lambda_cvar) for c in candidates)
            counts[w] += 1
            # incremental average update
            q_values[w] += (reward - q_values[w]) / counts[w]
        # pick best learned weights
        w_time, w_cost, w_rel = max(q_values.keys(), key=lambda k: q_values[k])

    # include RL candidate for scoring if present
    if 'rl_candidate' in locals() and rl_candidate:
        candidates.append(rl_candidate)
    for c in candidates:
        c["score"] = _multicriteria_score(c, w_time=w_time, w_cost=w_cost, w_rel=w_rel, lambda_cvar=lambda_cvar)
    candidates.sort(key=lambda d: d["score"], reverse=True)
    return candidates


def main():
    # demo
    origin = "NODE_000"; destination = "NODE_010"
    import pandas as pd
    cargo = {"weight_kg": 1200, "volume_m3": 12, "mode_pref": "combined", "created_at": pd.Timestamp("2024-01-05")}
    constraints = {"w_time": 1.0, "w_cost": 1.0, "w_rel": 1.0, "scenarios": 50, "deadline_hours": 200.0, "lambda_cvar": 0.1}
    routes = optimize_delivery(origin, destination, cargo, constraints)
    print(f"Found {len(routes)} route(s)")
    if routes:
        top = routes[0]
        print("Top route summary:", {k: top[k] for k in ["time_h", "cost", "reliability", "n_segments", "score"]})


if __name__ == "__main__":
    main()
# Optional RL routing policy
try:
    from .rl_routing import plan_route as rl_plan_route
except Exception:
    rl_plan_route = None  # type: ignore
