"""
Graph embeddings and node features generator.

Computes simple spectral (Laplacian) embeddings for nodes as a proxy for GNN embeddings
and saves them to node_embeddings.csv. Additionally, computes centrality-based node
features and merges with embeddings into node_features.csv to be consumed by models.
"""
import os
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx

# Support both module and script runs
try:
    from .utils import load_nodes_edges, build_graph_from_csvs
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import load_nodes_edges, build_graph_from_csvs


def spectral_embeddings(G: nx.Graph, dim: int = 8) -> pd.DataFrame:
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    # adjacency
    A = np.zeros((n, n), dtype=float)
    for u, v in G.edges():
        A[idx[u], idx[v]] = 1.0
        A[idx[v], idx[u]] = 1.0
    # degree matrix and Laplacian
    D = np.diag(A.sum(axis=1))
    L = D - A
    # eigen-decomposition (smallest non-zero eigenvectors)
    # eigh returns sorted eigenvalues; skip the first 1 (near 0)
    vals, vecs = np.linalg.eigh(L)
    # take next dim components
    start = 1 if n > 1 else 0
    end = min(start + dim, n)
    emb = vecs[:, start:end]
    cols = [f"emb_{i}" for i in range(emb.shape[1])]
    df = pd.DataFrame(emb, columns=cols)
    df.insert(0, "node_id", nodes)
    return df


def _centrality_features(G: nx.Graph) -> pd.DataFrame:
    nodes = list(G.nodes())
    # Degree (normalized), betweenness and closeness
    deg = dict(G.degree())
    n = max(len(nodes), 1)
    deg_norm = {k: (float(v) / max(n - 1, 1)) for k, v in deg.items()}
    try:
        btw = nx.betweenness_centrality(G, normalized=True)
    except Exception:
        btw = {k: 0.0 for k in nodes}
    try:
        clo = nx.closeness_centrality(G)
    except Exception:
        clo = {k: 0.0 for k in nodes}
    # Node attributes (type if present)
    types = {}
    for k, d in G.nodes(data=True):
        types[k] = str(d.get("type", "unknown"))
    df = pd.DataFrame({
        "node_id": nodes,
        "degree": [deg_norm.get(nid, 0.0) for nid in nodes],
        "betweenness": [btw.get(nid, 0.0) for nid in nodes],
        "closeness": [clo.get(nid, 0.0) for nid in nodes],
        "type": [types.get(nid, "unknown") for nid in nodes],
    })
    # One-hot for type (kept minimal; downstream can one-hot too)
    try:
        df = pd.get_dummies(df, columns=["type"], drop_first=False)
    except Exception:
        pass
    return df


def main(dim: int = 8, out_path: str = None):
    # default save to dataset/
    out_dir = "dataset"
    os.makedirs(out_dir, exist_ok=True)
    if out_path is None:
        out_path = os.path.join(out_dir, "node_embeddings.csv")
    nodes, edges = load_nodes_edges()
    G = build_graph_from_csvs(nodes, edges)
    emb_df = spectral_embeddings(G, dim=dim)
    emb_df.to_csv(out_path, index=False)
    print(f"Saved embeddings to {out_path} with shape {emb_df.shape}")

    # Build richer node_features.csv for downstream models
    feats_df = _centrality_features(G)
    # Merge with embeddings on node_id
    merged = feats_df.merge(emb_df, on="node_id", how="left")

    # Lightweight GNN-style mean aggregator over embeddings (GraphSAGE proxy)
    try:
        nodes = list(G.nodes())
        idx = {n: i for i, n in enumerate(nodes)}
        # Build adjacency list
        neigh = {n: list(G.neighbors(n)) for n in nodes}
        # Start from spectral embeddings
        E = merged[[c for c in merged.columns if c.startswith("emb_")]].values
        if E.size > 0:
            # one hop mean aggregation
            E1 = E.copy()
            for i, n in enumerate(nodes):
                ns = neigh.get(n, [])
                if ns:
                    E1[i, :] = (E[i, :] + np.mean([E[idx[nb], :] for nb in ns], axis=0)) / 2.0
            # second hop
            E2 = E1.copy()
            for i, n in enumerate(nodes):
                ns = neigh.get(n, [])
                if ns:
                    E2[i, :] = (E1[i, :] + np.mean([E1[idx[nb], :] for nb in ns], axis=0)) / 2.0
            # attach as sage_emb_*
            for j in range(E2.shape[1]):
                merged[f"sage_emb_{j}"] = E2[:, j]
    except Exception:
        pass
    feats_out = os.path.join(out_dir, "node_features.csv")
    merged.to_csv(feats_out, index=False)
    print(f"Saved node features to {feats_out} with shape {merged.shape}")


if __name__ == "__main__":
    main()
