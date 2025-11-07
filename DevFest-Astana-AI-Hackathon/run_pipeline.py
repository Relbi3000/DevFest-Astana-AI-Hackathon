import os
import shutil
import subprocess
import sys


DATASETS = "datasets"


def run(cmd: list[str], cwd: str | None = None) -> int:
    print(f"\n=== RUN: {' '.join(cmd)} ===")
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(f"ERROR (code {proc.returncode}) in: {' '.join(cmd)}")
    return proc.returncode


def ensure_datasets_dir():
    os.makedirs(DATASETS, exist_ok=True)


def move_to_datasets():
    files = [
        "nodes_test.csv",
        "edges_test.csv",
        "orders_test.csv",
        "distance_matrix.csv",
        "weather.csv",
        "order_segments.csv",
        "node_features.csv",
        "daily_orders.csv",
        "graph.gpickle",
        "graph.gexf",
        "node_embeddings.csv",
    ]
    for f in files:
        if os.path.exists(f):
            dst = os.path.join(DATASETS, f)
            try:
                shutil.move(f, dst)
                print(f"moved {f} -> {dst}")
            except Exception as e:
                print(f"skip move {f}: {e}")


def main():
    ensure_datasets_dir()

    # Helper: do we already have a dataset produced?
    def dataset_ready() -> bool:
        # accept either datasets/*_test.csv or dataset/*.csv
        have_nodes = any(os.path.exists(p) for p in [
            os.path.join(DATASETS, "nodes_test.csv"),
            os.path.join("dataset", "nodes.csv"),
        ])
        have_edges = any(os.path.exists(p) for p in [
            os.path.join(DATASETS, "edges_test.csv"),
            os.path.join("dataset", "edges.csv"),
        ])
        have_orders = any(os.path.exists(p) for p in [
            os.path.join(DATASETS, "orders_test.csv"),
            os.path.join("dataset", "orders.csv"),
        ])
        return have_nodes and have_edges and have_orders

    # 1) generate data (best-effort)
    gen_ran = False
    if os.path.exists("datagenerator4.py"):
        rc = run([sys.executable, "datagenerator4.py"])
        gen_ran = True
        if rc == 0:
            move_to_datasets()
    elif os.path.exists("datagenerator2.0b.py") and not dataset_ready():
        rc = run([sys.executable, "datagenerator2.0b.py"])
        gen_ran = True
        # datagenerator2.0b.py writes to ./dataset; no need to move
    else:
        print("Skip data generation (using existing dataset).")
    # If generation attempted but failed and dataset not present, abort
    if gen_ran and not dataset_ready():
        sys.exit(1)

    # 2) embeddings (optional)
    run([sys.executable, "-m", "ml.embeddings"])  # node_embeddings.csv

    # 3) GNN edge predictors (optional, requires torch)
    run([sys.executable, "-m", "ml.gnn_models"])  # models/edge_gnn.pt

    # 4) learn-to-route models (segment predictors)
    run([sys.executable, "-m", "ml.learn_to_route"])  # edge_time/edge_cost models

    # 5) demand forecast (+ full time-series Transformer)
    run([sys.executable, "-m", "ml.demand_forecast"])  # saves metrics and model

    # 6) transport classifier
    run([sys.executable, "-m", "ml.transport_classifier"])  # saves metrics and model

    # 7) RL policy for routing (optional)
    run([sys.executable, "-m", "ml.rl_routing"])  # trains DQN policy

    # 8) route optimizer demo
    run([sys.executable, "-m", "ml.route_optimizer"])  # prints top route

    print("\nPipeline completed. Check 'datasets/' and 'models/' for outputs.")


if __name__ == "__main__":
    main()
