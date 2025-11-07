"""
Transport mode classifier (enhanced).

- Reads orders_test.csv as labels (chosen_mode) and features from orders + nodes + weather.
- Engineers features: distance, deadline tightness, weight/volume, cargo class, seasonality, weather, route stats.
- Adds graph-based node features (centralities + spectral embeddings) for origin/destination, and their deltas.
- Handles class imbalance via class weights / sample weighting.
- Trains a classifier (LightGBM/XGBoost if installed, else RandomForest) and saves richer metrics.
- Optional: SHAP-based interpretability if `shap` is available, else permutation/feature importances.
"""
import os
import json
import pickle
from typing import Optional

import numpy as np
import pandas as pd

# Support both `python -m ml.transport_classifier` and direct `python ml/transport_classifier.py`
try:
    from .utils import load_orders, load_nodes_edges, load_weather, build_graph_from_csvs, euclid_km, load_node_features, resolve_path
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import load_orders, load_nodes_edges, load_weather, build_graph_from_csvs, euclid_km, load_node_features


def prepare_features() -> tuple[pd.DataFrame, pd.Series, list[str], pd.Series]:
    orders = load_orders()
    nodes, edges = load_nodes_edges()
    weather = load_weather()
    node_feats = load_node_features()

    # distance between origin and destination via coords
    ndx = nodes.set_index("node_id")
    def dist_row(row):
        n1 = ndx.loc[row["origin_id"]]
        n2 = ndx.loc[row["destination_id"]]
        return euclid_km(n1, n2)
    orders["dist_km"] = orders.apply(dist_row, axis=1)

    # deadline tightness
    orders["deadline_hours"] = (orders["required_delivery"] - orders["created_at"]).dt.total_seconds() / 3600.0

    # time features
    orders["dow"] = orders["created_at"].dt.weekday
    orders["month"] = orders["created_at"].dt.month
    orders["is_weekend"] = (orders["dow"] >= 5).astype(int)

    # weather join by date + origin (robust to alt columns)
    if weather is not None and len(weather) > 0:
        try:
            weather["date"] = pd.to_datetime(weather["date"]).dt.date
        except Exception:
            pass
        orders["date"] = orders["created_at"].dt.date
        w = weather.rename(columns={"node_id": "origin_id"}).copy()
        # normalize factor columns
        if "time_factor" not in w.columns:
            if "time_factor_road" in w.columns:
                w["time_factor"] = w["time_factor_road"]
            else:
                w["time_factor"] = 1.0
        if "cost_factor" not in w.columns:
            if "cost_factor_road" in w.columns:
                w["cost_factor"] = w["cost_factor_road"]
            else:
                w["cost_factor"] = 1.0
        orders = orders.merge(
            w[[c for c in ["date", "origin_id", "time_factor", "cost_factor"] if c in w.columns]],
            on=["date", "origin_id"],
            how="left",
        )
        # fill any remaining NaNs with neutral factors
        orders["time_factor"] = orders.get("time_factor", pd.Series(index=orders.index, dtype=float)).fillna(1.0)
        orders["cost_factor"] = orders.get("cost_factor", pd.Series(index=orders.index, dtype=float)).fillna(1.0)
    else:
        orders["time_factor"] = 1.0
        orders["cost_factor"] = 1.0

    # join node features for origin/destination if available
    if node_feats is not None and len(node_feats) > 0:
        nf = node_feats.set_index("node_id")
        orders = orders.join(nf, on="origin_id", rsuffix="_orig")
        orders = orders.join(nf, on="destination_id", rsuffix="_dest")
        # rename origin columns (no suffix) to _orig for clarity
        if "degree" in orders.columns:
            orders = orders.rename(columns={
                "degree": "degree_orig",
                "betweenness": "betweenness_orig",
                "closeness": "closeness_orig",
                "type": "type_orig",
            })
        # ensure origin embedding columns carry _orig suffix to avoid collision
        try:
            emb_cols = [c for c in orders.columns if c.startswith("emb_")]
            if emb_cols:
                orders = orders.rename(columns={c: f"{c}_orig" for c in emb_cols})
        except Exception:
            pass
        # types one-hot (only if raw type columns exist)
        for col in ["type_orig", "type_dest"]:
            if col in orders.columns:
                orders[col] = orders[col].astype(str)
        orders = pd.get_dummies(orders, columns=[c for c in ["type_orig", "type_dest"] if c in orders.columns], drop_first=False)

    # ensure n_segments exists (derive from route_nodes if missing)
    if "n_segments" not in orders.columns:
        def _seg_count(s):
            try:
                parts = [x for x in str(s).split(";") if len(x)]
                return max(len(parts) - 1, 0)
            except Exception:
                return 0
        if "route_nodes" in orders.columns:
            orders["n_segments"] = orders["route_nodes"].apply(_seg_count)
        else:
            orders["n_segments"] = 0

    # reliability_expected fallback
    if "reliability_expected" not in orders.columns:
        orders["reliability_expected"] = 0.95

    # one-hots for cargo_class and target label
    X = orders.copy()
    # label fallback: chosen_mode -> actual_mode -> preferred_mode
    if "chosen_mode" in X.columns:
        y = X["chosen_mode"].astype(str)
    elif "actual_mode" in X.columns:
        y = X["actual_mode"].astype(str)
    elif "preferred_mode" in X.columns:
        y = X["preferred_mode"].astype(str)
    else:
        # default to road to avoid crash, though training won't be meaningful
        y = pd.Series(["road"] * len(X))
    X = pd.get_dummies(X, columns=["cargo_class", "dow", "month"], drop_first=False)

    base_cols = [
        "dist_km", "deadline_hours", "weight_kg", "volume_m3",
        "is_weekend", "time_factor", "cost_factor",
        "n_segments", "reliability_expected",
    ]
    # add node feature columns if present
    node_cols = [
        c for c in X.columns
        if c.startswith("degree_") or c.startswith("betweenness_") or c.startswith("closeness_")
        or c.startswith("emb_") or c.startswith("type_orig_") or c.startswith("type_dest_")
        or c in ["degree_orig","betweenness_orig","closeness_orig","degree_dest","betweenness_dest","closeness_dest"]
        or c.startswith("diff_")
    ]
    cat_cols = [c for c in X.columns if c.startswith("cargo_class_") or c.startswith("dow_") or c.startswith("month_")]
    feature_cols = base_cols + node_cols + cat_cols

    # drop non-feature columns if present
    drop_cols = [
        "order_id", "origin_id", "destination_id", "created_at", "required_delivery", "earliest_pickup",
        "latest_pickup", "earliest_delivery", "latest_delivery", "status", "route_nodes", "route_modes",
        "route_edge_ids", "actual_time_h", "actual_cost", "lateness_h", "reliability_outcome", "date"
    ]
    for c in drop_cols:
        if c in X.columns:
            X = X.drop(columns=[c])

    # preserve created_at for temporal split
    created_at_series = orders["created_at"] if "created_at" in orders.columns else pd.Series([pd.Timestamp(0)] * len(orders))
    X = X[feature_cols].fillna(0.0)
    return X, y, feature_cols, created_at_series


def train_classifier(model_out: str = "models/transport_classifier.pkl") -> dict:
    os.makedirs("models", exist_ok=True)
    X, y, feature_cols, created_at = prepare_features()
    # time-based split by created_at
    try:
        order_idx = created_at.sort_values().index
        X = X.loc[order_idx].reset_index(drop=True)
        y = y.loc[order_idx].reset_index(drop=True)
    except Exception:
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
    n = len(X)
    split = int(n * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    # class imbalance handling via per-class weights
    vc = y_train.value_counts()
    total = float(len(y_train))
    n_classes = float(len(vc)) if len(vc) > 0 else 1.0
    class_weights = {cls: total / (n_classes * float(cnt)) for cls, cnt in vc.items()}
    sample_weight = y_train.map(class_weights).astype(float).values

    model = None
    rng_seed = 42
    try:
        import lightgbm as lgb
        use_gpu = bool(int(os.environ.get("USE_GPU", "0")))
        lgb_kwargs = dict(n_estimators=600, learning_rate=0.05, max_depth=-1, random_state=rng_seed, class_weight="balanced")
        if use_gpu:
            # device_type requires GPU-enabled LightGBM build
            lgb_kwargs["device_type"] = "gpu"
        model = lgb.LGBMClassifier(**lgb_kwargs)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    except Exception:
        try:
            import xgboost as xgb
            use_gpu = bool(int(os.environ.get("USE_GPU", "0")))
            tree_method = "gpu_hist" if use_gpu else "hist"
            model = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.8, random_state=rng_seed, reg_alpha=0.0, reg_lambda=1.0, tree_method=tree_method)
            model.fit(X_train, y_train, sample_weight=sample_weight)
        except Exception:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=rng_seed)
            try:
                model.fit(X_train, y_train, sample_weight=sample_weight)
            except Exception:
                model.fit(X_train, y_train)

    # predictions and metrics
    y_pred = model.predict(X_test)
    acc = float((y_pred == y_test).mean())
    # save model
    with open(model_out, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_cols": feature_cols,
            "classes_": getattr(model, "classes_", None)
        }, f)
    # extra metrics
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(y_test, y_pred, output_dict=True)
    macro_f1 = float(report.get("macro avg", {}).get("f1-score", 0.0))
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
    os.makedirs("models", exist_ok=True)
    # save confusion matrix
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in sorted(y.unique())], columns=[f"pred_{c}" for c in sorted(y.unique())])
    cm_df.to_csv("models/transport_confusion_matrix.csv")
    # save metrics JSON with class distribution (imbalance insight)
    class_dist = y.value_counts().to_dict()
    with open("models/transport_classifier_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": acc,
            "macro_f1": macro_f1,
            "report": report,
            "n_train": int(split),
            "n_test": int(n - split),
            "class_distribution": class_dist,
        }, f, ensure_ascii=False, indent=2)
    # save config
    with open("models/transport_classifier_config.json", "w", encoding="utf-8") as f:
        used_node_features = any(
            (
                c.startswith("degree_")
                or c.startswith("betweenness_")
                or c.startswith("closeness_")
                or c.startswith("emb_")
                or c.startswith("type_orig_")
                or c.startswith("type_dest_")
                or c.startswith("diff_")
                or c in [
                    "degree_orig",
                    "betweenness_orig",
                    "closeness_orig",
                    "degree_dest",
                    "betweenness_dest",
                    "closeness_dest",
                ]
            )
            for c in feature_cols
        )
        json.dump(
            {
                "feature_cols": feature_cols,
                "used_node_features": bool(used_node_features),
                "seed": rng_seed,
                "model_type": type(model).__name__,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    # Interpretability: SHAP if available, else feature_importances_
    try:
        import shap  # type: ignore
        sample_n = min(len(X_test), 1000)
        Xs = X_test.iloc[:sample_n]
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xs)
        import numpy as _np
        if isinstance(sv, list):
            sv_abs = _np.mean(_np.stack([_np.abs(s) for s in sv], axis=0), axis=0)
        else:
            sv_abs = _np.abs(sv)
        imp_vals = _np.mean(sv_abs, axis=0)
        shap_imp = {col: float(val) for col, val in zip(list(Xs.columns), imp_vals)}
        with open("models/transport_classifier_importance.json", "w", encoding="utf-8") as f:
            json.dump({"type": "shap_mean_abs", "importance": shap_imp}, f, ensure_ascii=False, indent=2)
    except Exception:
        try:
            fi = getattr(model, "feature_importances_", None)
            if fi is not None:
                imp = {col: float(val) for col, val in zip(list(X_train.columns), fi)}
                with open("models/transport_classifier_importance.json", "w", encoding="utf-8") as f:
                    json.dump({"type": "feature_importances_", "importance": imp}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return {"accuracy": acc, "n_train": int(split), "n_test": int(n - split)}


def main():
    metrics = train_classifier()
    print("Transport classifier baseline:", metrics)


if __name__ == "__main__":
    main()

