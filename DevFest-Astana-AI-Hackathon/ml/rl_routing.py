"""
Route planning via Reinforcement Learning (DQN).

Trains a DQN policy over graph nodes to select next-hop neighbors towards a
destination, optimizing a reward that trades off time, cost, and reliability.
Saves the learned policy into models/rl_policy.pt with the node index mapping.
Also exposes a plan_route(...) helper to generate a route using the policy.
"""
import os
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore

try:
    from .utils import load_nodes_edges, build_graph_from_csvs, load_weather
except ImportError:
    from utils import load_nodes_edges, build_graph_from_csvs, load_weather

# Optional advanced generators from route optimizer
try:
    from .route_optimizer import _quantum_simulated_paths, _reliability_greedy_path  # type: ignore
    from .route_optimizer import _load_graph as _load_graph_ro  # type: ignore
except Exception:
    _quantum_simulated_paths = None  # type: ignore
    _reliability_greedy_path = None  # type: ignore
    _load_graph_ro = None  # type: ignore

# Optional L2R models for reward shaping
def _load_edge_models():
    import pickle
    time_model = None; cost_model = None; feat_cols = None
    time_q = None; cost_q = None
    try:
        with open("models/edge_time_model.pkl", "rb") as f:
            t = pickle.load(f); time_model = t.get("model"); feat_cols = t.get("feature_cols")
        with open("models/edge_cost_model.pkl", "rb") as f:
            c = pickle.load(f); cost_model = c.get("model")
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
    return time_model, cost_model, feat_cols, time_q, cost_q

def _segment_features(d: Dict[str, float]) -> Dict[str, float]:
    mode = str(d.get("mode", "road"))
    return {
        "distance_km": float(d.get("distance_km", 0.0)),
        "base_time_h": float(d.get("base_time_h", 0.0)),
        "base_cost": float(d.get("base_cost", 0.0)),
        "max_weight_kg": float(d.get("max_weight_kg", 0.0)),
        "max_volume_m3": float(d.get("max_volume_m3", 0.0)),
        "reliability": float(d.get("reliability", 0.95)),
        "time_factor": 1.0,
        "cost_factor": 1.0,
        "weight_kg": 0.0,
        "volume_m3": 0.0,
        "mode_road": 1.0 if mode == "road" else 0.0,
        "mode_air": 1.0 if mode == "air" else 0.0,
        "mode_combined": 1.0 if mode == "combined" else 0.0,
    }


def _build_graph_and_index():
    nodes, edges = load_nodes_edges()
    G = build_graph_from_csvs(nodes, edges)
    node_ids = list(G.nodes())
    idx = {nid: i for i, nid in enumerate(node_ids)}
    adj: List[List[int]] = [[] for _ in node_ids]
    edge_attr: Dict[Tuple[int, int], Dict[str, float]] = {}
    for u, v, d in G.edges(data=True):
        ui = idx[u]; vi = idx[v]
        adj[ui].append(vi); adj[vi].append(ui)
        edge_attr[(ui, vi)] = {
            "base_time_h": float(d.get("base_time_h", 0.0)),
            "base_cost": float(d.get("base_cost", 0.0)),
            "reliability": float(d.get("reliability", 0.95)),
            "mode": str(d.get("mode", "road")),
        }
        edge_attr[(vi, ui)] = edge_attr[(ui, vi)]
    return G, idx, adj, edge_attr


class DQNPolicy(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int = 32):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, emb_dim)
        hidden = max(128, emb_dim * 4)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_nodes),
        )
    def forward(self, cur_idx: torch.Tensor, dst_idx: torch.Tensor) -> torch.Tensor:
        h_cur = self.emb(cur_idx)
        h_dst = self.emb(dst_idx)
        z = torch.cat([h_cur, h_dst], dim=-1)
        return self.mlp(z)  # Q-values over all nodes


def _edge_reward(
    d: Dict[str, float],
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    l2r_models=None,
    feat_cols=None,
    prev_mode: Optional[str] = None,
    transfer_penalty: float = 0.0,
    cvar_lambda: float = 0.0,
) -> float:
    mode = str(d.get("mode", "road"))
    base_t = float(d.get("base_time_h", 0.0))
    base_c = float(d.get("base_cost", 0.0))
    rel = float(d.get("reliability", 0.95))
    t_use, c_use = base_t, base_c
    rel_pen = 0.0
    if l2r_models is not None and feat_cols is not None:
        time_model, cost_model, time_q, cost_q = l2r_models
        try:
            import pandas as _pd
            X = _pd.DataFrame([_segment_features(d)])[list(feat_cols)].fillna(0.0)
            if time_model is not None:
                t_use = float(time_model.predict(X)[0])
            if cost_model is not None:
                c_use = float(cost_model.predict(X)[0])
            if cvar_lambda > 0.0 and time_q is not None and cost_q is not None:
                try:
                    tqm = time_q.get("0.9"); cqm = cost_q.get("0.9")
                    if tqm is not None and t_use > 0:
                        t_pen = float(tqm.predict(X)[0] / max(t_use, 1e-6) - 1.0)
                    else:
                        t_pen = 0.0
                    if cqm is not None and c_use > 0:
                        c_pen = float(cqm.predict(X)[0] / max(c_use, 1e-6) - 1.0)
                    else:
                        c_pen = 0.0
                    rel_pen = cvar_lambda * 0.5 * (max(t_pen, 0.0) + max(c_pen, 0.0))
                except Exception:
                    rel_pen = 0.0
        except Exception:
            pass
    reward = float(-(alpha * max(t_use, 0.0) + beta * max(c_use, 0.0)) + gamma * rel)
    if prev_mode is not None and mode != prev_mode:
        reward -= float(transfer_penalty)
    if rel < 0.8:
        reward -= (0.8 - rel) * 50.0
    reward -= float(rel_pen)
    return reward


def train_dqn(episodes: int = 2000, emb_dim: int = 32, *, alpha: float = 1.0, beta: float = 0.01, gamma_r: float = 2e2, transfer_penalty: float = 2.0, use_demos: bool = True, demo_scale: int = 3, use_quantum_demo: bool = True, use_reliability_demo: bool = True, cvar_lambda: float = 0.1) -> Optional[Dict[str, object]]:
    if torch is None:
        return None
    G, idx, adj, edge_attr = _build_graph_and_index()
    N = len(adj)
    if N == 0:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = DQNPolicy(num_nodes=N, emb_dim=emb_dim).to(device)
    target = DQNPolicy(num_nodes=N, emb_dim=emb_dim).to(device)
    target.load_state_dict(policy.state_dict())
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    gamma_disc = 0.95
    eps = 1.0
    eps_min = 0.05
    eps_decay = 0.995
    batch_size = 64
    replay: List[Tuple[int, int, int, float, int, bool]] = []  # (s_cur, s_dst, action, reward, s_next, done)

    rng = np.random.default_rng(123)
    node_ids = list(idx.keys())

    # prepare L2R models
    time_model, cost_model, feat_cols, time_q, cost_q = _load_edge_models()
    l2r = (time_model, cost_model, time_q, cost_q) if feat_cols is not None else None

    def step(s: int, dst: int, a: int, prev_mode: Optional[str]) -> Tuple[int, float, bool, str]:
        # invalid action -> no-op with heavy penalty
        if a not in adj[s]:
            return s, -5.0, False, prev_mode or "road"
        d = edge_attr[(s, a)]
        r = _edge_reward(d, alpha=alpha, beta=beta, gamma=gamma_r, l2r_models=l2r, feat_cols=feat_cols, prev_mode=prev_mode, transfer_penalty=transfer_penalty, cvar_lambda=cvar_lambda)
        done = (a == dst)
        return a, r + (500.0 if done else 0.0), done, str(d.get("mode", "road"))

    def _push_demo(path: List[int], dst: int):
        pm: Optional[str] = None
        for i in range(len(path) - 1):
            s = path[i]; a = path[i+1]
            ns, r, dn, pm = step(s, dst, a, pm)
            for _ in range(max(1, demo_scale)):
                replay.append((s, dst, a, r, ns, dn))

    import time as _time
    t0 = _time.time()
    import time as _time
    t0 = _time.time()
    for ep in range(episodes):
        s = int(rng.integers(0, N))
        dst = int(rng.integers(0, N))
        if dst == s:
            dst = (dst + 1) % N
        steps = 0
        done = False
        # inject demonstrations occasionally
        if use_demos and (ep % 10 == 0) and _load_graph_ro is not None:
            try:
                origin = list(idx.keys())[s]
                dest = list(idx.keys())[dst]
                Gfull = _load_graph_ro()
                if use_quantum_demo and _quantum_simulated_paths is not None:
                    qpaths = _quantum_simulated_paths(Gfull, origin, dest)
                    for qp in qpaths:
                        ids = [idx[n] for n in qp if n in idx]
                        if len(ids) >= 2 and ids[0] == s and ids[-1] == dst:
                            _push_demo(ids, dst)
                if use_reliability_demo and _reliability_greedy_path is not None:
                    rp = _reliability_greedy_path(Gfull, origin, dest)
                    if rp:
                        ids = [idx[n] for n in rp if n in idx]
                        if len(ids) >= 2 and ids[0] == s and ids[-1] == dst:
                            _push_demo(ids, dst)
            except Exception:
                pass
        prev_mode: Optional[str] = None
        while not done and steps < 4 * N:
            steps += 1
            # epsilon-greedy over neighbor actions
            if rng.random() < eps:
                if adj[s]:
                    a = int(adj[s][int(rng.integers(0, len(adj[s])))] )
                else:
                    a = s
            else:
                with torch.no_grad():
                    q = policy(torch.tensor([s], dtype=torch.long, device=device), torch.tensor([dst], dtype=torch.long, device=device)).cpu().numpy()[0]
                # mask non-neighbors to -inf
                mask = np.full_like(q, -1e9, dtype=np.float32)
                for nb in adj[s]:
                    mask[nb] = q[nb]
                a = int(np.argmax(mask))
            s_next, r, done, prev_mode = step(s, dst, a, prev_mode)
            replay.append((s, dst, a, r, s_next, done))
            if len(replay) > 50000:
                replay = replay[-50000:]
            s = s_next
            # train
            if len(replay) >= batch_size:
                batch = rng.choice(len(replay), size=batch_size, replace=False)
                sc, sd, ac, rc, sn, dn = zip(*[replay[i] for i in batch])
                sc_t = torch.tensor(sc, dtype=torch.long, device=device)
                sd_t = torch.tensor(sd, dtype=torch.long, device=device)
                ac_t = torch.tensor(ac, dtype=torch.long, device=device)
                rc_t = torch.tensor(rc, dtype=torch.float32, device=device)
                sn_t = torch.tensor(sn, dtype=torch.long, device=device)
                dn_t = torch.tensor(dn, dtype=torch.bool, device=device)
                q_pred = policy(sc_t, sd_t)
                q_sa = q_pred.gather(1, ac_t.view(-1, 1)).squeeze(1)
                with torch.no_grad():
                    q_next = target(sn_t, sd_t)
                    # mask non-neighbors for sn state
                    for i, sni in enumerate(sn):
                        mask = torch.full((q_next.shape[1],), -1e9, device=device)
                        for nb in adj[sni]:
                            mask[nb] = q_next[i, nb]
                        q_next[i] = mask
                    q_max = q_next.max(dim=1).values
                    tgt_val = rc_t + (~dn_t).float() * gamma_disc * q_max
                loss = nn.functional.mse_loss(q_sa, tgt_val)
                opt.zero_grad(); loss.backward(); opt.step()
        # dynamic epsilon
        eps = max(eps_min, eps * (eps_decay * (0.98 if steps < N/2 else 1.0)))
        if (ep + 1) % 50 == 0:
            target.load_state_dict(policy.state_dict())
            # progress + ETA
            done_frac = float(ep + 1) / float(episodes)
            elapsed = _time.time() - t0
            eta = (elapsed / max(done_frac, 1e-6)) * (1.0 - done_frac)
            print(f"[RL] progress: {int(done_frac*100)}% | elapsed {elapsed:.1f}s | ETA {eta:.1f}s")

    os.makedirs("models", exist_ok=True)
    torch.save({
        "state_dict": policy.state_dict(),
        "num_nodes": int(N),
        "emb_dim": int(emb_dim),
    }, "models/rl_policy.pt")
    # also save adjacency/nodes for debug
    return {"episodes": int(episodes), "num_nodes": int(N)}


def plan_route(origin: str, destination: str, max_steps: int = 200) -> List[str]:
    """Plans a route using the saved RL policy, falling back to greedy neighbors
    even if torch/policy is unavailable.
    """
    G, idx, adj, edge_attr = _build_graph_and_index()
    if origin not in idx or destination not in idx:
        return []
    s = idx[origin]; dst = idx[destination]
    path = [origin]
    visited = set([s])
    if torch is None or not os.path.exists("models/rl_policy.pt"):
        # fallback: greedy by base_time
        for _ in range(max_steps):
            if s == dst:
                break
            cand = adj[s]
            if not cand:
                break
            a = min(cand, key=lambda nb: float(edge_attr[(s, nb)].get("base_time_h", 1e9)))
            s = a
            nid = list(idx.keys())[list(idx.values()).index(a)]
            path.append(nid)
            if a in visited:
                break
            visited.add(a)
        return path if path[-1] == destination else []
    # DQN greedy
    payload = torch.load("models/rl_policy.pt", map_location="cpu")
    emb_dim = int(payload["emb_dim"])
    class _P(nn.Module):
        def __init__(self, N: int):
            super().__init__()
            self.emb = nn.Embedding(N, emb_dim)
            hidden = max(128, emb_dim * 4)
            self.mlp = nn.Sequential(
                nn.Linear(emb_dim * 2, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, N),
            )
        def forward(self, ci, di):
            h_cur = self.emb(ci); h_dst = self.emb(di)
            return self.mlp(torch.cat([h_cur, h_dst], dim=-1))
    pol = _P(int(payload["num_nodes"]))
    pol.load_state_dict(payload["state_dict"], strict=False)
    pol.eval()
    for _ in range(max_steps):
        if s == dst:
            break
        with torch.no_grad():
            q = pol(torch.tensor([s], dtype=torch.long), torch.tensor([dst], dtype=torch.long)).numpy()[0]
        mask = np.full_like(q, -1e9, dtype=np.float32)
        for nb in adj[s]:
            mask[nb] = q[nb]
        a = int(np.argmax(mask))
        s = a
        nid = list(idx.keys())[list(idx.values()).index(a)]
        path.append(nid)
        if a in visited:
            break
        visited.add(a)
    return path if path[-1] == destination else []


def main():
    # Train RL policy (best-effort; skips if torch missing)
    res = train_dqn()
    if res is None:
        print("RL training skipped (torch unavailable or small graph)")
    else:
        print("RL policy trained:", res)


if __name__ == "__main__":
    main()
