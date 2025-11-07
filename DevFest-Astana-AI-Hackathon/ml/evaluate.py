"""
Evaluation utilities:

- Demand MAPE tracker: reads models/demand_forecast_predictions.csv and recomputes MAPE.
- Transport confusion matrix/metrics viewer: recomputes metrics from orders_test.csv split.
"""
import os
import json
import numpy as np
import pandas as pd

from typing import Dict


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), 1e-6, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def eval_demand(pred_path: str = "models/demand_forecast_predictions.csv") -> Dict:
    if not os.path.exists(pred_path):
        raise FileNotFoundError(pred_path)
    df = pd.read_csv(pred_path, parse_dates=["date"])
    score = mape(df["y_true"].values, df["y_pred"].values)
    return {"mape": score, "n": int(len(df))}


def eval_transport(orders_path: str = "orders_test.csv") -> Dict:
    from sklearn.metrics import classification_report, confusion_matrix
    df = pd.read_csv(orders_path, parse_dates=["created_at"])  # load orders
    # prepare split
    df = df.sort_values("created_at").reset_index(drop=True)
    split = int(len(df) * 0.8)
    test = df.iloc[split:]
    y_true = test["chosen_mode"].astype(str).values
    # if predictions existed, we could load them; for now we recompute a trivial baseline (most frequent)
    most_freq = df["chosen_mode"].astype(str).mode().iat[0]
    y_pred = np.array([most_freq] * len(test))
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=sorted(df["chosen_mode"].astype(str).unique()))
    # save artifacts
    os.makedirs("models", exist_ok=True)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in sorted(df["chosen_mode"].astype(str).unique())], columns=[f"pred_{c}" for c in sorted(df["chosen_mode"].astype(str).unique())])
    cm_df.to_csv("models/transport_baseline_confusion_matrix.csv")
    with open("models/transport_baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return {"accuracy": float((y_pred == y_true).mean())}


def main():
    try:
        dm = eval_demand()
        print("Demand MAPE tracker:", dm)
    except Exception as e:
        print("Demand eval skipped:", e)
    try:
        tm = eval_transport()
        print("Transport baseline metrics:", tm)
    except Exception as e:
        print("Transport eval skipped:", e)


if __name__ == "__main__":
    main()

