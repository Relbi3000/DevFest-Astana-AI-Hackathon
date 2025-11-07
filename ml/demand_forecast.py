"""
Demand forecasting (enhanced baseline).

- Reads orders_test.csv to compute daily aggregates (or uses daily_orders.csv if present).
- Joins weather and time features.
- Trains a model (LightGBM/XGBoost if installed, else LinearRegression) to predict next-day n_orders.
- Adds: rolling backtesting, optional quantile forecasts (P10/P50/P90) when LightGBM is available,
  and interpretability artifacts (SHAP if installed, else feature importances).
"""
import os
import json
import pickle
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

# Optional: full time-series Transformer training/eval
try:
    from .ts_transformer import train_ts_transformer, evaluate_ts_transformer_or_none
except Exception:
    def train_ts_transformer(*args, **kwargs):
        return None
    def evaluate_ts_transformer_or_none(*args, **kwargs):
        return None

# Support both `python -m ml.demand_forecast` and direct `python ml/demand_forecast.py`
try:
    from .utils import load_orders, load_weather, ensure_daily_orders, add_time_features, join_weather_daily
except ImportError:  # no package context
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import load_orders, load_weather, ensure_daily_orders, add_time_features, join_weather_daily


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), 1e-6, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def train_baseline(daily: pd.DataFrame, horizon: int = 1, model_out: str = "models/demand_forecast.pkl") -> dict:
    os.makedirs("models", exist_ok=True)
    # simple supervised framing: predict n_orders[t+h] from features at t
    df = daily.copy()
    # base time features
    df = add_time_features(df, date_col="date")
    # seasonal sin/cos features
    df["day_of_year"] = df["date"].dt.dayofyear
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.0)
    # encode non-numeric columns (e.g., aggregated weather label)
    if "weather_mode" in df.columns:
        df = pd.get_dummies(df, columns=["weather_mode"], prefix="weather", drop_first=False)
    # lag features
    df = df.sort_values("date").reset_index(drop=True)
    # lags
    for lag in [1, 7, 14, 28]:
        df[f"n_orders_lag_{lag}"] = df["n_orders"].shift(lag)
    # moving averages (shifted to avoid leakage)
    for win in [7, 14, 28]:
        df[f"n_orders_ma_{win}"] = df["n_orders"].rolling(window=win, min_periods=win).mean().shift(1)
    # target
    df["target"] = df["n_orders"].shift(-horizon)
    df = df.dropna().reset_index(drop=True)
    # split by time (last 20% as test)
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]
    # keep only numeric features to avoid dtype issues
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ["target", "n_orders"]]

    X_train, y_train = train[feature_cols].values, train["target"].values
    X_test, y_test = test[feature_cols].values, test["target"].values

    # Seed for reproducibility (also used by optional torch model)
    rng_seed = 42

    # Optional: lightweight Transformer regressor over feature tokens
    y_pred_transformer = None
    mape_transformer: Optional[float] = None
    tf_artifacts: Dict[str, object] = {}
    try:
        import torch
        import torch.nn as nn
        torch.manual_seed(rng_seed)

        in_dim = X_train.shape[1]
        d_model = min(64, max(16, in_dim))
        # ensure nhead divides d_model
        nhead = 1
        for h in [8, 4, 2, 1]:
            if d_model % h == 0:
                nhead = h
                break

        class TinyFeatureTransformer(nn.Module):
            def __init__(self, in_dim: int, d_model: int, nhead: int, num_layers: int = 2):
                super().__init__()
                self.token_proj = nn.Linear(1, d_model)
                enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
                self.out = nn.Linear(d_model, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: (B, F) or (B, F, 1)
                if x.dim() == 2:
                    x = x.unsqueeze(-1)
                # per-feature token projection -> (B, F, d_model)
                z = self.token_proj(x)
                z = self.encoder(z)  # (B, F, d_model)
                z = z.mean(dim=1)    # global mean pool over tokens -> (B, d_model)
                y = self.out(z)      # (B, 1)
                return y.squeeze(-1)

        model_tf = TinyFeatureTransformer(in_dim=in_dim, d_model=d_model, nhead=nhead, num_layers=2)
        opt = torch.optim.Adam(model_tf.parameters(), lr=1e-3)
        loss_fn = nn.L1Loss()

        XT = torch.tensor(X_train, dtype=torch.float32)  # (B, F)
        yT = torch.tensor(y_train, dtype=torch.float32)
        model_tf.train()
        for _ in range(12):  # a few quick epochs
            opt.zero_grad()
            z = model_tf(XT)
            loss = loss_fn(z, yT)
            loss.backward(); opt.step()
        model_tf.eval()
        with torch.no_grad():
            XTe = torch.tensor(X_test, dtype=torch.float32)
            y_pred_transformer = model_tf(XTe).cpu().numpy()
            mape_transformer = mape(y_test, y_pred_transformer)

        # keep artifacts in case the transformer is chosen as best
        tf_artifacts = {
            "tf_state_dict": model_tf.state_dict(),
            "tf_params": {"in_dim": int(in_dim), "d_model": int(d_model), "nhead": int(nhead), "num_layers": 2},
        }
    except Exception:
        y_pred_transformer = None
        mape_transformer = None

    model = None
    try:
        # try LightGBM
        import lightgbm as lgb
        model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, max_depth=-1, random_state=rng_seed)
        model.fit(X_train, y_train)
    except Exception:
        try:
            # fallback to XGBoost
            import xgboost as xgb
            model = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.8, random_state=rng_seed, reg_alpha=0.0, reg_lambda=1.0)
            model.fit(X_train, y_train)
        except Exception:
            # final fallback: LinearRegression from sklearn
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mape_boosted = mape(y_test, y_pred)
    chosen_model = "boosted"
    # choose better of transformer and boosted/linear model if transformer available
    if y_pred_transformer is not None and mape_transformer is not None:
        if mape_transformer <= mape_boosted:
            y_pred = y_pred_transformer
            chosen_model = "transformer"
    score_mape = mape(y_test, y_pred)
    # save predictions for MAPE tracker
    pred_df = pd.DataFrame({
        "date": test["date"].values,
        "y_true": y_test,
        "y_pred": y_pred,
    })
    os.makedirs("models", exist_ok=True)
    pred_df.to_csv("models/demand_forecast_predictions.csv", index=False)

    payload: Dict[str, object] = {
        "model": model,
        "feature_cols": feature_cols,
        "horizon": horizon,
        "best_model_type": chosen_model,
    }
    if chosen_model == "transformer" and tf_artifacts:
        # store transformer weights/params for possible reuse
        payload.update(tf_artifacts)

    # Optional: train quantile models if LightGBM is available
    quantiles_available = False
    try:
        import lightgbm as lgb  # type: ignore
        q_models = {}
        for alpha in [0.1, 0.5, 0.9]:
            qm = lgb.LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=600, learning_rate=0.05, max_depth=-1, random_state=rng_seed)
            qm.fit(X_train, y_train)
            q_models[str(alpha)] = qm
        payload["quantile_models"] = q_models
        quantiles_available = True
        # save quantile predictions for the test set
        q_pred = {f"q{int(alpha*100)}": q_models[str(alpha)].predict(X_test) for alpha in [0.1, 0.5, 0.9]}
        pred_df = pred_df.copy()
        for k, v in q_pred.items():
            pred_df[k] = v
    except Exception:
        pass

    with open(model_out, "wb") as f:
        pickle.dump(payload, f)

    # save metrics JSON
    # Backtesting via rolling origin split
    backtest_scores: List[float] = []
    try:
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=max(2, min(5, len(df)//20)))
        for tr_idx, te_idx in tscv.split(df):
            tr, te = df.iloc[tr_idx], df.iloc[te_idx]
            Xtr, ytr = tr[feature_cols].values, tr["target"].values
            Xte, yte = te[feature_cols].values, te["target"].values
            try:
                import lightgbm as lgb
                mdl = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=-1, random_state=rng_seed)
                mdl.fit(Xtr, ytr)
            except Exception:
                try:
                    import xgboost as xgb
                    mdl = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.8, random_state=rng_seed)
                    mdl.fit(Xtr, ytr)
                except Exception:
                    from sklearn.linear_model import LinearRegression
                    mdl = LinearRegression().fit(Xtr, ytr)
            yp = mdl.predict(Xte)
            backtest_scores.append(mape(yte, yp))
    except Exception:
        pass

    metrics_payload = {"mape": score_mape, "n_train": int(len(train)), "n_test": int(len(test))}
    metrics_payload["mape_boosted"] = float(mape_boosted)
    if mape_transformer is not None:
        metrics_payload["mape_transformer"] = float(mape_transformer)
        metrics_payload["chosen_model"] = chosen_model
    if backtest_scores:
        metrics_payload.update({
            "backtest_mapes": backtest_scores,
            "backtest_mape_mean": float(np.mean(backtest_scores)),
            "backtest_mape_std": float(np.std(backtest_scores)),
        })
    metrics_payload["quantiles_available"] = quantiles_available
    with open("models/demand_forecast_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    # save config
    with open("models/demand_forecast_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "horizon": horizon,
            "feature_cols": feature_cols,
            "lags": [1,7,14,28],
            "moving_averages": [7,14,28],
            "seasonal": ["dow_sin","dow_cos","doy_sin","doy_cos"],
            "seed": rng_seed,
            "model_type": type(model).__name__,
            "transformer_available": bool(y_pred_transformer is not None),
            "best_model_type": chosen_model,
        }, f, ensure_ascii=False, indent=2)

    # Interpretability artifacts
    try:
        import shap  # type: ignore
        # use a small sample to keep speed reasonable
        sample_n = min(len(train), 1000)
        Xs = train[feature_cols].iloc[:sample_n]
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xs)
        import numpy as _np
        imp_vals = _np.mean(_np.abs(sv), axis=0)
        shap_imp = {col: float(val) for col, val in zip(feature_cols, imp_vals)}
        with open("models/demand_forecast_importance.json", "w", encoding="utf-8") as f:
            json.dump({"type": "shap_mean_abs", "importance": shap_imp}, f, ensure_ascii=False, indent=2)
    except Exception:
        try:
            fi = getattr(model, "feature_importances_", None)
            if fi is not None:
                imp = {col: float(val) for col, val in zip(feature_cols, fi)}
                with open("models/demand_forecast_importance.json", "w", encoding="utf-8") as f:
                    json.dump({"type": "feature_importances_", "importance": imp}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return {"mape": score_mape, "n_train": len(train), "n_test": len(test)}


def main():
    # build daily aggregates
    orders = load_orders()
    if os.path.exists("daily_orders.csv"):
        daily = pd.read_csv("daily_orders.csv", parse_dates=["date"])  # optional precomputed
    else:
        daily = ensure_daily_orders(orders)
        daily["date"] = pd.to_datetime(daily["date"])  # ensure dtype
    weather = load_weather()
    daily_joined = join_weather_daily(daily, weather)
    # Train full sequence Transformer first (if torch available)
    try:
        ts_info = train_ts_transformer(daily_joined, horizon=1, window=28)
        if ts_info is not None:
            print("Trained Time-series Transformer:", ts_info)
    except Exception as _e:
        pass
    metrics = train_baseline(daily_joined)
    print("Demand forecast baseline:", metrics)
    # Compare/evaluate transformer and write metric into metrics JSON
    try:
        m_ts = evaluate_ts_transformer_or_none(daily_joined)
        if m_ts is not None:
            try:
                with open("models/demand_forecast_metrics.json", "r", encoding="utf-8") as f:
                    cur = json.load(f)
            except Exception:
                cur = {}
            cur["mape_ts_transformer"] = float(m_ts)
            with open("models/demand_forecast_metrics.json", "w", encoding="utf-8") as f:
                json.dump(cur, f, ensure_ascii=False, indent=2)
            print("Time-series Transformer val MAPE:", m_ts)
    except Exception:
        pass


if __name__ == "__main__":
    main()
