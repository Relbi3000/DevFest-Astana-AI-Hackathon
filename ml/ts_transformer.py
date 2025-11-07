"""
Time-series Transformer training for daily demand forecasting.

Trains a proper sequence-to-one Transformer on sliding windows over the daily
aggregates and saves the model to models/demand_ts_transformer.pt. Provides a
helper to evaluate and report validation MAPE on the temporal holdout.
"""
import os
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), 1e-6, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _make_sliding_windows(
    daily: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "n_orders",
    window: int = 28,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    df = daily.sort_values("date").reset_index(drop=True)
    Xs: List[np.ndarray] = []
    ys: List[float] = []
    vals = df[feature_cols].values.astype(float)
    tgt = df[target_col].values.astype(float)
    for i in range(window, len(df) - horizon + 1):
        Xs.append(vals[i - window:i, :])
        ys.append(tgt[i + horizon - 1])
    if not Xs:
        return np.zeros((0, window, len(feature_cols))), np.zeros((0,))
    return np.stack(Xs, axis=0), np.array(ys)


def _prepare_numeric_features(daily: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    from .utils import add_time_features
    df = daily.copy()
    df = add_time_features(df, date_col="date")
    df["day_of_year"] = df["date"].dt.dayofyear
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.0)
    # keep only numeric features, target is n_orders
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != "n_orders"]
    return df, feature_cols


def _build_model(in_dim: int, d_model: int, nhead: int, num_layers: int = 3):
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 512):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))
        def forward(self, x):
            T = x.size(1)
            return x + self.pe[:, :T, :]

    class TimeSeriesTransformer(nn.Module):
        def __init__(self, in_dim: int, d_model: int, nhead: int, num_layers: int = 3):
            super().__init__()
            self.input_proj = nn.Linear(in_dim, d_model)
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.posenc = PositionalEncoding(d_model)
            self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        def forward(self, x):
            z = self.input_proj(x)
            z = self.posenc(z)
            z = self.encoder(z)
            z = z[:, -1, :]
            return self.out(z).squeeze(-1)

    return TimeSeriesTransformer(in_dim=in_dim, d_model=d_model, nhead=nhead, num_layers=num_layers)


def train_ts_transformer(daily: pd.DataFrame, horizon: int = 1, window: int = 28) -> Optional[Dict[str, object]]:
    if torch is None:
        return None
    df, feature_cols = _prepare_numeric_features(daily)
    X, y = _make_sliding_windows(df, feature_cols, target_col="n_orders", window=window, horizon=horizon)
    if len(X) < 10:
        return None
    split = int(len(X) * 0.8)
    Xtr, Ytr = X[:split], y[:split]
    Xte, Yte = X[split:], y[split:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = min(128, max(32, X.shape[2]))
    nhead = 1
    for h in [8, 4, 2, 1]:
        if d_model % h == 0:
            nhead = h
            break
    model = _build_model(in_dim=X.shape[2], d_model=d_model, nhead=nhead, num_layers=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()
    tr_ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Ytr, dtype=torch.float32))
    te_ds = TensorDataset(torch.tensor(Xte, dtype=torch.float32), torch.tensor(Yte, dtype=torch.float32))
    tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=64)
    best_val = float('inf')
    best_state = None
    import time as _time
    t0 = _time.time()
    total_epochs = 30
    for ep in range(total_epochs):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); pred = model(xb)
            loss = loss_fn(pred, yb); loss.backward(); opt.step()
        # quick val MAPE
        model.eval(); preds = []; trues = []
        with torch.no_grad():
            for xb, yb in te_dl:
                preds.append(model(xb.to(device)).cpu().numpy()); trues.append(yb.numpy())
        yp = np.concatenate(preds) if preds else np.array([])
        yt = np.concatenate(trues) if trues else np.array([])
        val_mape = mape(yt, yp) if len(yp) else float('inf')
        if val_mape < best_val:
            best_val = val_mape
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        # progress + ETA
        done_frac = float(ep + 1) / float(total_epochs)
        elapsed = _time.time() - t0
        eta = (elapsed / max(done_frac, 1e-6)) * (1.0 - done_frac)
        print(f"[TS-Transformer] epoch {ep+1}/{total_epochs} | val MAPE={val_mape:.2f} | elapsed {elapsed:.1f}s | ETA {eta:.1f}s")
    if best_state is None:
        return None
    os.makedirs("models", exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "in_dim": int(X.shape[2]),
        "d_model": int(d_model),
        "nhead": int(nhead),
        "window": int(window),
        "horizon": int(horizon),
        "feature_cols": feature_cols,
    }, "models/demand_ts_transformer.pt")
    return {"val_mape": float(best_val), "window": int(window), "feature_cols": feature_cols}


def evaluate_ts_transformer_or_none(daily: pd.DataFrame) -> Optional[float]:
    """Loads the saved transformer and computes validation MAPE on temporal
    holdout. Returns None if model or torch is unavailable.
    """
    if torch is None:
        return None
    try:
        payload = torch.load("models/demand_ts_transformer.pt", map_location="cpu")
    except Exception:
        return None
    from .utils import add_time_features
    base = add_time_features(daily.copy(), date_col="date")
    base["day_of_year"] = base["date"].dt.dayofyear
    base["dow_sin"] = np.sin(2 * np.pi * base["dow"] / 7.0)
    base["dow_cos"] = np.cos(2 * np.pi * base["dow"] / 7.0)
    base["doy_sin"] = np.sin(2 * np.pi * base["day_of_year"] / 365.0)
    base["doy_cos"] = np.cos(2 * np.pi * base["day_of_year"] / 365.0)
    fcols: List[str] = list(payload.get("feature_cols", []))
    window = int(payload.get("window", 28))
    X, y = _make_sliding_windows(base, fcols, target_col="n_orders", window=window, horizon=1)
    if len(X) < 5:
        return None
    split = int(len(X) * 0.8)
    Xte, yte = X[split:], y[split:]
    # Rebuild model skeleton
    in_dim = int(payload["in_dim"]); d_model = int(payload["d_model"]); nhead = int(payload["nhead"]) 
    model = _build_model(in_dim, d_model, nhead)
    try:
        model.load_state_dict(payload["state_dict"], strict=False)
    except Exception:
        return None
    model.eval()
    with torch.no_grad():
        yp = model(torch.tensor(Xte, dtype=torch.float32)).numpy()
    return mape(yte, yp)
