from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

# Optional deps; fall back gracefully if unavailable
try:
    import osmnx as ox  # type: ignore
except Exception:  # pragma: no cover
    ox = None  # type: ignore

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    from faker import Faker  # type: ignore
except Exception:  # pragma: no cover
    Faker = None  # type: ignore


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return 6371.0 * c


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class DatasetArtifacts:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    orders: pd.DataFrame
    distance_matrix: pd.DataFrame
    time_matrix: pd.DataFrame
    cost_matrix: pd.DataFrame
    weather: pd.DataFrame
    scenarios: pd.DataFrame
    metadata: Dict[str, Any]


@dataclass
class GeneratorConfig:
    osm_mode: str = "bbox"
    bbox: Tuple[float, float, float, float] = (45.0, 55.0, 60.0, 70.0)
    place_name: str = ""
    n_nodes: int = 36
    history_days: int = 150
    mean_orders_per_day: int = 65
    cache_dir: Path = Path("./.cache")
    output_dir: Path = Path("./dataset")
    seed: int = 42
    w1: float = 0.35
    w2: float = 0.25
    w3: float = 0.25
    w4: float = 0.15
    anomalies_share: float = 0.1
    export_profile: str = "standard"
    schema_version: str = "3.0"
    od_distance_decay: float = 1.1
    transfer_radius_km: float = 75.0
    road_peak_hours: Tuple[Tuple[int, int], Tuple[int, int]] = ((7, 10), (17, 20))
    weekend_peak_multiplier: float = 0.85
    anchor_days_offset: int = 0
    minimal_reliability: float = 0.45


class OSMRoadSampler:
    def __init__(self, cfg: GeneratorConfig):
        self.cfg = cfg
        self.cache_dir = cfg.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(cfg.seed)
        self.faker = Faker() if Faker is not None else None
        self.graph = None
        self.available = False

    def _cache_key(self) -> Path:
        if self.cfg.osm_mode == "place" and self.cfg.place_name:
            name = self.cfg.place_name.replace(" ", "_")
            return self.cache_dir / f"osm_place_{name}.graphml"
        s, n, w, e = self.cfg.bbox
        return self.cache_dir / f"osm_bbox_{s:.2f}_{n:.2f}_{w:.2f}_{e:.2f}.graphml"

    def try_load_osm(self) -> None:
        if ox is None:
            self.available = False
            return
        cache_path = self._cache_key()
        try:
            if cache_path.exists():
                self.graph = ox.load_graphml(cache_path)
                self.available = True
                return
            if self.cfg.osm_mode == "place" and self.cfg.place_name:
                graph = ox.graph_from_place(
                    self.cfg.place_name, network_type="drive", simplify=True
                )
            else:
                s, n, w, e = self.cfg.bbox
                graph = ox.graph_from_bbox(
                    north=n, south=s, east=e, west=w, network_type="drive", simplify=True
                )
            ox.save_graphml(graph, cache_path)
            self.graph = graph
            self.available = True
        except Exception:
            self.available = False
            self.graph = None

    def sample_nodes(self, n_nodes: int) -> pd.DataFrame:
        self.try_load_osm()
        nodes: List[Dict[str, Any]] = []
        node_types = self._assign_node_types(n_nodes)
        if self.available and self.graph is not None:
            all_nodes = list(self.graph.nodes(data=True))
            picks = self.rng.choice(len(all_nodes), size=n_nodes, replace=False)
            for i, pick in enumerate(picks):
                osm_id, attrs = all_nodes[int(pick)]
                nodes.append(
                    {
                        "node_id": f"NODE_{i:03d}",
                        "osm_node_id": str(osm_id),
                        "name": (self.faker.company() if self.faker else f"Node {i}"),
                        "type": node_types[i],
                        "lat": float(attrs.get("y", self._rand_lat())),
                        "lon": float(attrs.get("x", self._rand_lon())),
                        "is_active": True,
                    }
                )
        else:
            for i in range(n_nodes):
                nodes.append(
                    {
                        "node_id": f"NODE_{i:03d}",
                        "osm_node_id": "",
                        "name": (self.faker.company() if self.faker else f"Node {i}"),
                        "type": node_types[i],
                        "lat": self._rand_lat(),
                        "lon": self._rand_lon(),
                        "is_active": True,
                    }
                )
        return pd.DataFrame(nodes)

    def build_edges(self, nodes_df: pd.DataFrame) -> pd.DataFrame:
        if self.available and self.graph is not None:
            return self._edges_from_osm(nodes_df)
        return self._edges_synthetic(nodes_df)

    def _edges_from_osm(self, nodes_df: pd.DataFrame) -> pd.DataFrame:
        osm_map = dict(zip(nodes_df["node_id"], nodes_df["osm_node_id"]))
        node_types = nodes_df.set_index("node_id")["type"].to_dict()
        edges: List[Dict[str, Any]] = []
        node_ids = nodes_df["node_id"].tolist()
        for i, a in enumerate(node_ids):
            dists: List[Tuple[float, str]] = []
            for j, b in enumerate(node_ids):
                if i == j:
                    continue
                try:
                    dist_m = nx.shortest_path_length(
                        self.graph, source=int(osm_map[a]), target=int(osm_map[b]), weight="length"
                    )
                    dists.append((float(dist_m) / 1000.0, b))
                except Exception:
                    pass
            dists.sort(key=lambda x: x[0])
            for dist_km, b in dists[:6]:
                if a < b:
                    edges.extend(self._make_edges(a, b, dist_km=dist_km, node_types=node_types))
        return pd.DataFrame(edges)

    def _edges_synthetic(self, nodes_df: pd.DataFrame) -> pd.DataFrame:
        lat_map = nodes_df.set_index("node_id")["lat"].to_dict()
        lon_map = nodes_df.set_index("node_id")["lon"].to_dict()
        node_types = nodes_df.set_index("node_id")["type"].to_dict()
        edges: List[Dict[str, Any]] = []
        node_ids = nodes_df["node_id"].tolist()
        for i, a in enumerate(node_ids):
            dists: List[Tuple[float, str]] = []
            for j, b in enumerate(node_ids):
                if i == j:
                    continue
                dist_km = haversine(lat_map[a], lon_map[a], lat_map[b], lon_map[b])
                dists.append((dist_km, b))
            dists.sort(key=lambda x: x[0])
            for dist_km, b in dists[:6]:
                if a < b:
                    edges.extend(self._make_edges(a, b, dist_km=dist_km, node_types=node_types))
        return pd.DataFrame(edges)

    def _assign_node_types(self, n_nodes: int) -> List[str]:
        n_airports = max(2, int(0.2 * n_nodes))
        n_hubs = max(4, int(0.4 * n_nodes))
        n_warehouses = n_nodes - n_airports - n_hubs
        types = ["airport"] * n_airports + ["hub"] * n_hubs + ["warehouse"] * n_warehouses
        self.rng.shuffle(types)
        return types

    def _make_edges(
        self,
        a: str,
        b: str,
        dist_km: float,
        *,
        node_types: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        edges: List[Dict[str, Any]] = []
        speed_road = float(self.rng.uniform(50, 72))
        cost_per_km_road = float(self.rng.uniform(32, 56))
        base_time_road = max(0.4, dist_km / speed_road)
        edges.append(
            {
                "edge_id": f"EDGE_ROAD_{a}_{b}",
                "from_node": a,
                "to_node": b,
                "mode": "road",
                "distance_km": dist_km,
                "base_time_h": base_time_road,
                "base_cost": dist_km * cost_per_km_road + 420.0,
                "max_weight_kg": float(self.rng.uniform(12000, 26000)),
                "max_volume_m3": float(self.rng.uniform(90, 360)),
                "reliability": float(self.rng.uniform(0.86, 0.97)),
                "co2_kg_per_km": 0.18,
                "schedule_frequency_h": 0.0,
                "cutoff_hours": 0.0,
                "is_active": True,
            }
        )

        air_likelihood = 0.6 if "airport" in (node_types[a], node_types[b]) else 0.25
        if self.rng.uniform() < air_likelihood:
            speed_air = 720.0
            cruise = max(1.0, dist_km / speed_air)
            edges.append(
                {
                    "edge_id": f"EDGE_AIR_{a}_{b}",
                    "from_node": a,
                    "to_node": b,
                    "mode": "air",
                    "distance_km": dist_km * float(self.rng.uniform(1.02, 1.08)),
                    "base_time_h": cruise + 1.8,
                    "base_cost": 9000.0 + dist_km * float(self.rng.uniform(140, 220)),
                    "max_weight_kg": float(self.rng.uniform(7000, 16000)),
                    "max_volume_m3": float(self.rng.uniform(30, 120)),
                    "reliability": float(self.rng.uniform(0.88, 0.99)),
                    "co2_kg_per_km": 0.62,
                    "schedule_frequency_h": float(self.rng.choice([4, 6, 8, 12])),
                    "cutoff_hours": float(self.rng.uniform(2.0, 6.0)),
                    "is_active": True,
                }
            )

        return edges

    def _rand_lat(self) -> float:
        s, n, _, _ = self.cfg.bbox
        return float(self.rng.uniform(s, n))

    def _rand_lon(self) -> float:
        _, _, w, e = self.cfg.bbox
        return float(self.rng.uniform(w, e))


class OpenSkyConnector:
    API_URL = "https://opensky-network.org/api/flights/all"

    def __init__(self, cache_dir: Path, seed: int = 42):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)

    def fetch_schedule(self, start: datetime, end: datetime, max_rows: int = 5000) -> pd.DataFrame:
        if requests is None:
            return pd.DataFrame()
        cache_path = self.cache_dir / f"opensky_{start:%Y%m%d}_{end:%Y%m%d}.csv"
        if cache_path.exists():
            try:
                return pd.read_csv(cache_path)
            except Exception:
                pass
        try:
            params = {
                "begin": int(start.replace(tzinfo=timezone.utc).timestamp()),
                "end": int(end.replace(tzinfo=timezone.utc).timestamp()),
            }
            resp = requests.get(self.API_URL, params=params, timeout=60)
            resp.raise_for_status()
            flights = pd.DataFrame(resp.json())
            if len(flights) > max_rows:
                flights = flights.sample(max_rows, random_state=int(self.rng.integers(0, 1_000_000)))
            flights.to_csv(cache_path, index=False)
            return flights
        except Exception:
            return pd.DataFrame()


class WeatherSynthesizer:
    EVENT_CATALOG: Dict[str, Dict[str, Any]] = {
        "clear": {
            "temp": (18, 8),
            "visibility": (16, 3),
            "wind": (8, 5),
            "precip": (0.2, 0.2),
            "time_mult": {"road": 1.0, "air": 1.0, "transfer": 1.0},
            "cost_mult": {"road": 1.0, "air": 1.0, "transfer": 1.0},
            "reliability_penalty": {"road": 0.0, "air": 0.0},
        },
        "rain": {
            "temp": (12, 6),
            "visibility": (8, 3),
            "wind": (16, 6),
            "precip": (4, 2),
            "time_mult": {"road": 1.12, "air": 1.05, "transfer": 1.06},
            "cost_mult": {"road": 1.05, "air": 1.08, "transfer": 1.04},
            "reliability_penalty": {"road": 0.02, "air": 0.015},
        },
        "snow": {
            "temp": (-6, 5),
            "visibility": (4, 1.5),
            "wind": (18, 8),
            "precip": (6, 3),
            "time_mult": {"road": 1.4, "air": 1.18, "transfer": 1.12},
            "cost_mult": {"road": 1.16, "air": 1.12, "transfer": 1.09},
            "reliability_penalty": {"road": 0.07, "air": 0.09},
        },
        "storm": {
            "temp": (9, 7),
            "visibility": (3, 1.2),
            "wind": (44, 12),
            "precip": (12, 4),
            "time_mult": {"road": 1.32, "air": 1.4, "transfer": 1.22},
            "cost_mult": {"road": 1.12, "air": 1.3, "transfer": 1.18},
            "reliability_penalty": {"road": 0.12, "air": 0.22},
        },
        "fog": {
            "temp": (3, 4),
            "visibility": (1.6, 0.8),
            "wind": (4, 2),
            "precip": (0.4, 0.3),
            "time_mult": {"road": 1.18, "air": 1.26, "transfer": 1.08},
            "cost_mult": {"road": 1.04, "air": 1.18, "transfer": 1.05},
            "reliability_penalty": {"road": 0.05, "air": 0.12},
        },
        "ice": {
            "temp": (-9, 3),
            "visibility": (6, 2),
            "wind": (9, 4),
            "precip": (1.3, 0.6),
            "time_mult": {"road": 1.35, "air": 1.09, "transfer": 1.18},
            "cost_mult": {"road": 1.16, "air": 1.07, "transfer": 1.12},
            "reliability_penalty": {"road": 0.1, "air": 0.05},
        },
        "wind": {
            "temp": (14, 6),
            "visibility": (10, 2),
            "wind": (36, 10),
            "precip": (0.6, 0.4),
            "time_mult": {"road": 1.06, "air": 1.16, "transfer": 1.05},
            "cost_mult": {"road": 1.03, "air": 1.11, "transfer": 1.02},
            "reliability_penalty": {"road": 0.03, "air": 0.09},
        },
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(self, nodes: pd.DataFrame, start: datetime, days: int) -> pd.DataFrame:
        recs: List[Dict[str, Any]] = []
        clusters = self._cluster_nodes(nodes)
        cluster_map = {nid: cid for nid, cid in clusters}
        for d in range(days):
            date = start + timedelta(days=d)
            month = date.month
            cluster_events = self._sample_cluster_events(month, set(cluster_map.values()))
            for _, row in nodes.iterrows():
                nid = row["node_id"]
                cid = cluster_map[nid]
                base_event = cluster_events[cid]
                severity = float(self.rng.uniform(0.8, 1.2))
                catalog = self.EVENT_CATALOG[base_event]
                temp = float(self.rng.normal(catalog["temp"][0], catalog["temp"][1]))
                visibility = max(0.2, float(self.rng.normal(catalog["visibility"][0], catalog["visibility"][1])))
                wind = abs(float(self.rng.normal(catalog["wind"][0], catalog["wind"][1])))
                precip = max(0.0, float(self.rng.normal(catalog["precip"][0], catalog["precip"][1])))
                recs.append(
                    {
                        "date": date,
                        "node_id": nid,
                        "cluster": cid,
                        "event": base_event,
                        "severity": severity,
                        "temperature_c": temp,
                        "visibility_km": visibility,
                        "wind_kts": wind,
                        "precip_mm": precip,
                        "time_factor_road": catalog["time_mult"]["road"] * severity,
                        "time_factor_air": catalog["time_mult"]["air"] * severity,
                        "time_factor_transfer": catalog["time_mult"]["transfer"] * severity,
                        "cost_factor_road": catalog["cost_mult"]["road"] * clamp(severity, 0.7, 1.6),
                        "cost_factor_air": catalog["cost_mult"]["air"] * clamp(severity, 0.7, 1.6),
                        "cost_factor_transfer": catalog["cost_mult"]["transfer"] * clamp(severity, 0.7, 1.6),
                        "reliability_penalty_road": catalog["reliability_penalty"]["road"] * severity,
                        "reliability_penalty_air": catalog["reliability_penalty"]["air"] * severity,
                    }
                )
        return pd.DataFrame(recs)

    def _cluster_nodes(self, nodes: pd.DataFrame, n_lat_bins: int = 3, n_lon_bins: int = 3):
        lat_min, lat_max = nodes["lat"].min(), nodes["lat"].max()
        lon_min, lon_max = nodes["lon"].min(), nodes["lon"].max()
        lat_span = max(0.0001, lat_max - lat_min)
        lon_span = max(0.0001, lon_max - lon_min)
        clusters: List[Tuple[str, int]] = []
        for _, row in nodes.iterrows():
            lat_idx = int((row["lat"] - lat_min) / lat_span * n_lat_bins)
            lon_idx = int((row["lon"] - lon_min) / lon_span * n_lon_bins)
            lat_idx = clamp(lat_idx, 0, n_lat_bins - 1)
            lon_idx = clamp(lon_idx, 0, n_lon_bins - 1)
            clusters.append((row["node_id"], int(lat_idx * n_lon_bins + lon_idx)))
        return clusters

    def _sample_cluster_events(self, month: int, cluster_ids: Iterable[int]) -> Dict[int, str]:
        if month in (12, 1, 2):
            probs = {"snow": 0.28, "ice": 0.18, "storm": 0.08, "clear": 0.22, "fog": 0.12, "wind": 0.07, "rain": 0.05}
        elif month in (3, 4, 5):
            probs = {"snow": 0.08, "ice": 0.05, "storm": 0.09, "clear": 0.3, "fog": 0.12, "wind": 0.12, "rain": 0.24}
        elif month in (6, 7, 8):
            probs = {"clear": 0.34, "storm": 0.12, "rain": 0.22, "wind": 0.12, "fog": 0.06, "snow": 0.0, "ice": 0.0}
        else:
            probs = {"clear": 0.26, "storm": 0.1, "rain": 0.28, "wind": 0.13, "fog": 0.09, "snow": 0.08, "ice": 0.06}
        catalog = list(self.EVENT_CATALOG.keys())
        weights = np.array([probs.get(evt, 0.01) for evt in catalog], dtype=float)
        if weights.sum() <= 0:
            weights[:] = 1.0
        weights /= weights.sum()
        assignments: Dict[int, str] = {}
        for cid in cluster_ids:
            assignments[cid] = self.rng.choice(catalog, p=weights)
        return assignments


class ScenarioEngine:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        weather: pd.DataFrame,
    ) -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        if edges.empty:
            return pd.DataFrame(records)

        weather_by_date = weather.groupby("date")
        edges_by_mode = edges.groupby("mode")
        start_date = weather["date"].min()
        end_date = weather["date"].max()

        sampled_edges = edges.sample(
            frac=0.12, random_state=int(self.rng.integers(0, 1_000_000))
        )
        for _, edge_row in sampled_edges.iterrows():
            span_days = int(self.rng.integers(1, 4))
            start = start_date + timedelta(
                days=int(
                    self.rng.integers(0, max(1, (end_date - start_date).days - span_days))
                )
            )
            records.append(
                {
                    "event_id": f"SCN_MAINT_{edge_row['edge_id']}",
                    "event_type": "maintenance",
                    "target_type": "edge",
                    "target_id": edge_row["edge_id"],
                    "start": start,
                    "end": start + timedelta(days=span_days),
                    "severity": float(self.rng.uniform(0.4, 0.8)),
                    "payload": json.dumps(
                        {
                            "time_mult": self.rng.uniform(1.2, 1.6),
                            "cost_mult": self.rng.uniform(1.1, 1.4),
                        }
                    ),
                }
            )

        for date, subframe in weather_by_date:
            if "storm" in set(subframe["event"]) or "snow" in set(subframe["event"]):
                mode = "air" if "storm" in set(subframe["event"]) else "road"
                if mode not in edges_by_mode.groups:
                    continue
                pick = edges_by_mode.get_group(mode).sample(
                    n=max(1, int(edges_by_mode.get_group(mode).shape[0] * 0.12)),
                    random_state=int(self.rng.integers(0, 1_000_000)),
                )
                for _, edge_row in pick.iterrows():
                    duration = int(self.rng.integers(1, 3))
                    records.append(
                        {
                            "event_id": f"SCN_WE_{edge_row['edge_id']}_{date:%Y%m%d}",
                            "event_type": "weather_shutdown" if mode == "air" else "road_block",
                            "target_type": "edge",
                            "target_id": edge_row["edge_id"],
                            "start": date,
                            "end": date + timedelta(days=duration),
                            "severity": float(self.rng.uniform(0.6, 1.0)),
                            "payload": json.dumps({"closed": True}),
                        }
                    )

        if "road" in edges_by_mode.groups:
            shock_start = start_date + timedelta(days=int(self.rng.integers(15, 45)))
            records.append(
                {
                    "event_id": "SCN_FUEL_SPIKE",
                    "event_type": "cost_shock",
                    "target_type": "mode",
                    "target_id": "road",
                    "start": shock_start,
                    "end": shock_start + timedelta(days=int(self.rng.integers(7, 18))),
                    "severity": float(self.rng.uniform(0.12, 0.3)),
                    "payload": json.dumps({"cost_additive_per_km": self.rng.uniform(6, 12)}),
                }
            )

        airports = nodes[nodes["type"] == "airport"]
        if not airports.empty:
            airport = airports.sample(
                n=1, random_state=int(self.rng.integers(0, 1_000_000))
            ).iloc[0]
            start = start_date + timedelta(days=int(self.rng.integers(30, 60)))
            records.append(
                {
                    "event_id": f"SCN_AP_SHUT_{airport['node_id']}",
                    "event_type": "node_shutdown",
                    "target_type": "node",
                    "target_id": airport["node_id"],
                    "start": start,
                    "end": start + timedelta(days=int(self.rng.integers(2, 6))),
                    "severity": 1.0,
                    "payload": json.dumps({"is_active": False}),
                }
            )

        return pd.DataFrame(records)


@dataclass
class DataGenerator:
    cfg: GeneratorConfig
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.cfg.seed)
        self.osm = OSMRoadSampler(self.cfg)
        self.opensky = OpenSkyConnector(self.cfg.cache_dir, seed=self.cfg.seed + 1)
        self.weather_gen = WeatherSynthesizer(seed=self.cfg.seed + 2)
        self.scenario_engine = ScenarioEngine(seed=self.cfg.seed + 3)
        self._edge_lookup: Dict[str, Tuple[str, str, str]] = {}

    def build_dataset(self) -> DatasetArtifacts:
        nodes = self._enrich_nodes(self.osm.sample_nodes(self.cfg.n_nodes))
        base_edges = self.osm.build_edges(nodes)
        edges = self._add_transfer_edges(base_edges, nodes)
        graph = self._build_graph(nodes, edges)
        start = self._history_start()
        weather = self.weather_gen.generate(nodes, start, self.cfg.history_days)
        scenarios = self.scenario_engine.generate(nodes, edges, weather)
        self._apply_scenarios_to_graph(graph, scenarios)
        air_schedule = self._derive_air_schedule(graph, start)
        orders = self._simulate_orders(graph, nodes, weather, scenarios, air_schedule)
        distance_matrix, time_matrix, cost_matrix = self._distance_time_cost_matrices(graph, nodes)
        validations = self._validate_generation(orders, nodes, edges)
        meta = {
            "generated_at": datetime.utcnow().isoformat(),
            "anchor_date": self._history_anchor().isoformat(),
            "history_days": self.cfg.history_days,
            "schema_version": self.cfg.schema_version,
            "export_profile": self.cfg.export_profile,
            "quality_checks": json.dumps(validations),
            "config": json.dumps(self._config_metadata()),
        }
        return DatasetArtifacts(
            nodes=nodes,
            edges=edges,
            orders=orders,
            distance_matrix=distance_matrix,
            time_matrix=time_matrix,
            cost_matrix=cost_matrix,
            weather=weather,
            scenarios=scenarios,
            metadata=meta,
        )

    def export(self, artifacts: DatasetArtifacts) -> None:
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        node_cols = [
            "node_id",
            "name",
            "type",
            "lat",
            "lon",
            "operating_start_hour",
            "operating_end_hour",
            "cutoff_hours",
            "transfer_profiles",
            "service_time_min",
            "is_active",
        ]
        artifacts.nodes.loc[:, node_cols].to_csv(self.cfg.output_dir / "nodes.csv", index=False)

        edge_cols = [
            "edge_id",
            "from_node",
            "to_node",
            "mode",
            "distance_km",
            "base_time_h",
            "base_cost",
            "max_weight_kg",
            "max_volume_m3",
            "reliability",
            "co2_kg_per_km",
            "schedule_frequency_h",
            "cutoff_hours",
            "is_active",
        ]
        artifacts.edges.loc[:, edge_cols].to_csv(self.cfg.output_dir / "edges.csv", index=False)
        artifacts.distance_matrix.to_csv(self.cfg.output_dir / "distance_matrix.csv", index=False)
        artifacts.time_matrix.to_csv(self.cfg.output_dir / "time_matrix.csv", index=False)
        artifacts.cost_matrix.to_csv(self.cfg.output_dir / "cost_matrix.csv", index=False)
        artifacts.weather.to_csv(self.cfg.output_dir / "weather.csv", index=False)
        artifacts.scenarios.to_csv(self.cfg.output_dir / "scenarios.csv", index=False)

        orders = artifacts.orders
        minimal_cols = [
            "order_id",
            "origin_id",
            "destination_id",
            "created_at",
            "required_delivery",
            "status",
            "preferred_mode",
            "actual_time_h",
            "actual_cost",
            "lateness_h",
            "reliability_expected",
            "reliability_outcome",
            "score",
        ]
        standard_extra = [
            "weight_kg",
            "volume_m3",
            "cargo_class",
            "priority",
            "actual_mode",
            "dow",
            "month",
            "is_holiday",
            "origin_weather_event",
            "destination_weather_event",
            "weather_severity",
            "route_nodes",
            "route_modes",
            "capacity_ok",
            "co2_kg",
            "service_time_h",
            "transfer_time_h",
            "waiting_time_h",
            "fuel_cost",
            "handling_cost",
            "terminal_cost",
            "penalty_cost",
            "air_share",
            "congestion_factor",
            "road_only_time_h",
            "road_only_cost",
            "road_only_score",
            "air_only_time_h",
            "air_only_cost",
            "air_only_score",
            "combined_time_h",
            "combined_cost",
            "combined_score",
            "dataset_split",
        ]
        if self.cfg.export_profile == "minimal":
            orders_out = orders.loc[:, [c for c in minimal_cols if c in orders.columns]]
        elif self.cfg.export_profile == "standard":
            cols = minimal_cols + [c for c in standard_extra if c in orders.columns]
            orders_out = orders.loc[:, cols]
        else:
            orders_out = orders
        orders_out.to_csv(self.cfg.output_dir / "orders.csv", index=False)

    def _history_anchor(self) -> datetime:
        base = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        if self.cfg.anchor_days_offset:
            base = base + timedelta(days=self.cfg.anchor_days_offset)
        return base

    def _history_start(self) -> datetime:
        return self._history_anchor() - timedelta(days=self.cfg.history_days)

    def _enrich_nodes(self, nodes: pd.DataFrame) -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        for _, row in nodes.iterrows():
            node_type = row["type"]
            if node_type == "airport":
                operating = (4, 24)
                cutoff = 3.0
                transfer_profile = {"road->air": {"time_h": 1.6, "cost": 2200}}
                service_min = 65
            elif node_type == "hub":
                operating = (5, 23)
                cutoff = 2.0
                transfer_profile = {
                    "road->road": {"time_h": 0.6, "cost": 480},
                    "road->air": {"time_h": 1.4, "cost": 1800},
                }
                service_min = 48
            else:
                operating = (6, 22)
                cutoff = 1.0
                transfer_profile = {"road->road": {"time_h": 0.5, "cost": 360}}
                service_min = 32
            rec = row.to_dict()
            rec.update(
                {
                    "operating_start_hour": operating[0],
                    "operating_end_hour": operating[1],
                    "cutoff_hours": cutoff,
                    "transfer_profiles": json.dumps(transfer_profile),
                    "service_time_min": service_min,
                }
            )
            records.append(rec)
        return pd.DataFrame(records)

    def _add_transfer_edges(self, edges: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
        transfers: List[Dict[str, Any]] = []
        lat_map = nodes.set_index("node_id")["lat"].to_dict()
        lon_map = nodes.set_index("node_id")["lon"].to_dict()
        type_map = nodes.set_index("node_id")["type"].to_dict()
        node_ids = nodes["node_id"].tolist()
        for i, a in enumerate(node_ids):
            for b in node_ids[i + 1 :]:
                pair = tuple(sorted((type_map[a], type_map[b])))
                if pair not in {("airport", "hub"), ("hub", "warehouse"), ("airport", "warehouse")}:
                    continue
                dist = haversine(lat_map[a], lon_map[a], lat_map[b], lon_map[b])
                if dist > self.cfg.transfer_radius_km:
                    continue
                transfers.append(
                    {
                        "edge_id": f"EDGE_TRANSFER_{a}_{b}",
                        "from_node": a,
                        "to_node": b,
                        "mode": "transfer",
                        "distance_km": dist,
                        "base_time_h": 0.6 + dist / 80.0,
                        "base_cost": 800.0 + dist * 12.0,
                        "max_weight_kg": 60000.0,
                        "max_volume_m3": 600.0,
                        "reliability": 0.995,
                        "co2_kg_per_km": 0.02,
                        "schedule_frequency_h": 0.0,
                        "cutoff_hours": 0.0,
                        "is_active": True,
                    }
                )
        if transfers:
            edges = pd.concat([edges, pd.DataFrame(transfers)], ignore_index=True)
        return edges

    def _build_graph(self, nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.MultiGraph:
        graph = nx.MultiGraph()
        for _, r in nodes.iterrows():
            if not bool(r.get("is_active", True)):
                continue
            attrs = r.to_dict()
            attrs["transfer_profiles"] = json.loads(attrs.get("transfer_profiles", "{}"))
            graph.add_node(r["node_id"], **attrs)
        self._edge_lookup.clear()
        for _, e in edges.iterrows():
            if not bool(e.get("is_active", True)):
                continue
            u, v = e["from_node"], e["to_node"]
            if u not in graph.nodes or v not in graph.nodes:
                continue
            attrs = e.to_dict()
            attrs.setdefault("events", [])
            key = attrs["mode"]
            if graph.has_edge(u, v, key=key):
                key = f"{attrs['mode']}_{attrs['edge_id']}"
            graph.add_edge(u, v, key=key, **attrs)
            self._edge_lookup[attrs["edge_id"]] = (u, v, key)
        return graph

    def _apply_scenarios_to_graph(self, graph: nx.MultiGraph, scenarios: pd.DataFrame) -> None:
        if scenarios.empty:
            return
        for _, row in scenarios.iterrows():
            try:
                payload = json.loads(row["payload"])
            except Exception:
                payload = {}
            event = {
                "event_id": row["event_id"],
                "event_type": row["event_type"],
                "start": row["start"],
                "end": row["end"],
                "severity": row.get("severity", 1.0),
                "payload": payload,
            }
            target_type = row["target_type"]
            if target_type == "edge":
                lookup = self._edge_lookup.get(row["target_id"])
                if lookup:
                    u, v, key = lookup
                    events = list(graph[u][v][key].get("events", []))
                    events.append(event)
                    graph[u][v][key]["events"] = events
            elif target_type == "node":
                node_id = row["target_id"]
                if node_id in graph.nodes:
                    events = list(graph.nodes[node_id].get("events", []))
                    events.append(event)
                    graph.nodes[node_id]["events"] = events
            elif target_type == "mode":
                for u, v, key, data in graph.edges(keys=True, data=True):
                    if data.get("mode") == row["target_id"]:
                        events = list(data.get("events", []))
                        events.append(event)
                        graph[u][v][key]["events"] = events

    def _derive_air_schedule(self, graph: nx.MultiGraph, start: datetime) -> Dict[str, float]:
        flights = self.opensky.fetch_schedule(start, start + timedelta(days=7))
        schedule: Dict[str, float] = {}
        if flights.empty:
            return schedule
        counts = Counter()
        for _, row in flights.iterrows():
            dep = row.get("estDepartureAirport")
            arr = row.get("estArrivalAirport")
            if not isinstance(dep, str) or not isinstance(arr, str):
                continue
            key = f"{dep}_{arr}"
            counts[key] += 1
        for key, cnt in counts.items():
            schedule[key] = max(2.0, 168.0 / cnt)
        return schedule

    def _simulate_orders(
        self,
        graph: nx.MultiGraph,
        nodes: pd.DataFrame,
        weather: pd.DataFrame,
        scenarios: pd.DataFrame,
        air_schedule: Dict[str, float],
    ) -> pd.DataFrame:
        start = self._history_start()
        node_ids = nodes["node_id"].tolist()
        type_map = nodes.set_index("node_id")["type"].to_dict()
        lat_map = nodes.set_index("node_id")["lat"].to_dict()
        lon_map = nodes.set_index("node_id")["lon"].to_dict()
        weather_index = self._weather_index(weather)
        node_events = self._node_events_index(graph)
        mode_events = self._mode_events_index(graph)
        od_weights = self._build_od_weights(node_ids, type_map, lat_map, lon_map)
        od_cumulative = self._od_cumulative(od_weights)
        orders: List[Dict[str, Any]] = []
        for day in range(self.cfg.history_days):
            date = start + timedelta(days=day)
            dow = date.weekday()
            holiday_flag, _ = self._holiday_info(date)
            season_mult = self._seasonal_multiplier(date.month)
            weather_bias = self._daily_weather_bias(weather, date)
            lam = (
                self.cfg.mean_orders_per_day
                * self._weekday_multiplier(dow)
                * season_mult
                * weather_bias
            )
            n_orders = max(4, int(self.rng.poisson(lam)))
            for _ in range(n_orders):
                origin, dest = self._sample_od_pair(od_cumulative)
                if origin == dest:
                    continue
                created_at = self._random_time(date)
                cargo_class = self._cargo_class()
                weight_kg, volume_m3 = self._cargo_size(cargo_class)
                required_delivery = created_at + timedelta(hours=float(self.rng.uniform(36, 168)))
                route = self._route(
                    graph,
                    origin,
                    dest,
                    cargo_class,
                    weight_kg,
                    volume_m3,
                    created_at,
                    weather_index,
                    node_events,
                    mode_events,
                    air_schedule,
                )
                if route is None:
                    continue
                variants = self._route_variants(
                    graph,
                    origin,
                    dest,
                    cargo_class,
                    weight_kg,
                    volume_m3,
                    created_at,
                    weather_index,
                    node_events,
                    mode_events,
                    air_schedule,
                )
                reliability_expected = clamp(route["reliability"], self.cfg.minimal_reliability, 0.999)
                weather_penalty = route["risk_profile"]["weather_delay_prob"]
                scenario_penalty = route["risk_profile"]["scenario_delay_prob"]
                cancel_prob = clamp(route["risk_profile"]["cancel_prob"], 0.0, 0.8)
                delay_prob = clamp(weather_penalty + scenario_penalty, 0.0, 0.85)
                random_draw = self.rng.uniform()
                if random_draw < cancel_prob:
                    status = "cancelled"
                    reliability_outcome = 0.0
                    actual_time = 0.0
                    actual_cost = 0.0
                    lateness_h = float((created_at - required_delivery).total_seconds() / 3600.0)
                else:
                    reliability_outcome = 1.0 if self.rng.uniform() < reliability_expected else 0.0
                    status = "delivered"
                    base_time = route["time_h"]
                    base_cost = route["cost"]
                    delay_multiplier = 1.0
                    extra_cost_multiplier = 1.0
                    if self.rng.uniform() < delay_prob:
                        status = "delayed"
                        delay_multiplier += float(self.rng.uniform(0.15, 0.65))
                        extra_cost_multiplier += float(self.rng.uniform(0.05, 0.25))
                    actual_time = max(
                        1.0,
                        base_time * delay_multiplier * float(self.rng.normal(1.0, 0.08)),
                    )
                    actual_cost = max(
                        500.0,
                        base_cost * extra_cost_multiplier * float(self.rng.normal(1.0, 0.12)),
                    )
                    lateness_h = (
                        created_at + timedelta(hours=actual_time) - required_delivery
                    ).total_seconds() / 3600.0
                score = self._score(actual_time, actual_cost, reliability_expected, route["capacity_ok"])
                orders.append(
                    {
                        "order_id": f"ORD_{len(orders):07d}",
                        "origin_id": origin,
                        "destination_id": dest,
                        "created_at": created_at,
                        "required_delivery": required_delivery,
                        "earliest_pickup": created_at,
                        "latest_pickup": created_at + timedelta(hours=float(self.rng.uniform(1, 4))),
                        "earliest_delivery": created_at + timedelta(hours=route["time_h"] * 0.85),
                        "latest_delivery": required_delivery,
                        "weight_kg": weight_kg,
                        "volume_m3": volume_m3,
                        "cargo_class": cargo_class,
                        "priority": self._priority(cargo_class),
                        "status": status,
                        "dow": dow,
                        "month": date.month,
                        "is_holiday": int(holiday_flag),
                        "origin_weather_event": route["weather_summary"]["origin_event"],
                        "destination_weather_event": route["weather_summary"]["destination_event"],
                        "weather_severity": route["weather_summary"]["severity"],
                        "preferred_mode": route["preferred_mode"],
                        "route_nodes": ";".join(route["nodes"]),
                        "route_modes": ";".join(route["modes"]),
                        "reliability_expected": reliability_expected,
                        "capacity_ok": route["capacity_ok"],
                        "actual_time_h": actual_time,
                        "actual_cost": actual_cost,
                        "lateness_h": lateness_h,
                        "reliability_outcome": reliability_outcome,
                        "score": score,
                        "actual_mode": route["actual_mode"],
                        "co2_kg": route["co2_kg"],
                        "service_time_h": route["service_time_h"],
                        "transfer_time_h": route["transfer_time_h"],
                        "waiting_time_h": route["waiting_time_h"],
                        "fuel_cost": route["cost_breakdown"]["fuel"],
                        "handling_cost": route["cost_breakdown"]["handling"],
                        "terminal_cost": route["cost_breakdown"]["terminal"],
                        "penalty_cost": route["cost_breakdown"]["penalty"],
                        "air_share": route["share"]["air"],
                        "congestion_factor": route["congestion_factor"],
                        "road_only_time_h": variants["road"]["time"],
                        "road_only_cost": variants["road"]["cost"],
                        "road_only_score": variants["road"]["score"],
                        "air_only_time_h": variants["air"]["time"],
                        "air_only_cost": variants["air"]["cost"],
                        "air_only_score": variants["air"]["score"],
                        "combined_time_h": variants["combined"]["time"],
                        "combined_cost": variants["combined"]["cost"],
                        "combined_score": variants["combined"]["score"],
                        "dataset_split": self._split_for_date(created_at),
                    }
                )
        return pd.DataFrame(orders)

    def _distance_time_cost_matrices(
        self, graph: nx.MultiGraph, nodes: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        ids = sorted(nodes["node_id"].tolist())
        dist_rows: List[Dict[str, Any]] = []
        time_rows: List[Dict[str, Any]] = []
        cost_rows: List[Dict[str, Any]] = []
        for i, src in enumerate(ids):
            for dst in ids[i + 1 :]:
                dist = self._multi_weight_shortest(graph, src, dst, "distance_km")
                time_val = self._multi_weight_shortest(graph, src, dst, "base_time_h")
                cost_val = self._multi_weight_shortest(graph, src, dst, "base_cost")
                if dist is None:
                    dist = haversine(
                        float(nodes.loc[nodes["node_id"] == src, "lat"].iloc[0]),
                        float(nodes.loc[nodes["node_id"] == src, "lon"].iloc[0]),
                        float(nodes.loc[nodes["node_id"] == dst, "lat"].iloc[0]),
                        float(nodes.loc[nodes["node_id"] == dst, "lon"].iloc[0]),
                    )
                if time_val is None or cost_val is None:
                    est_time, est_cost = self._estimate_time_cost(dist, "road", 12000, 120)
                    time_val = est_time if time_val is None else time_val
                    cost_val = est_cost if cost_val is None else cost_val
                dist_rows.append({"from_node": src, "to_node": dst, "distance_km": float(dist)})
                time_rows.append({"from_node": src, "to_node": dst, "time_h": float(time_val)})
                cost_rows.append({"from_node": src, "to_node": dst, "cost": float(cost_val)})
        return pd.DataFrame(dist_rows), pd.DataFrame(time_rows), pd.DataFrame(cost_rows)

    def _multi_weight_shortest(
        self, graph: nx.MultiGraph, origin: str, dest: str, weight: str
    ) -> Optional[float]:
        try:
            return float(nx.shortest_path_length(graph, origin, dest, weight=weight))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _route(
        self,
        graph: nx.MultiGraph,
        origin: str,
        dest: str,
        cargo_class: str,
        weight: float,
        volume: float,
        created_at: datetime,
        weather_index: Dict[Tuple[str, datetime], Dict[str, Any]],
        node_events: Dict[str, List[Dict[str, Any]]],
        mode_events: Dict[str, List[Dict[str, Any]]],
        air_schedule: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        preferred_mode = self._preferred_mode(cargo_class)
        attempts = [
            (self._allowed_modes_for_preference(preferred_mode), True),
            ({"road", "transfer"}, True),
            ({"air", "transfer"}, True),
            (None, False),
        ]
        for allowed_modes, capacity_matters in attempts:
            filtered = self._filtered_graph(graph, allowed_modes, weight, volume, created_at)
            if origin not in filtered.nodes or dest not in filtered.nodes:
                continue
            try:
                path = nx.shortest_path(filtered, origin, dest, weight=lambda u, v, d: d.get("base_time_h", 1.0))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            stats = self._evaluate_path(
                graph,
                filtered,
                path,
                weight,
                volume,
                created_at,
                weather_index,
                node_events,
                mode_events,
                air_schedule,
            )
            if stats["capacity_ok"] or not capacity_matters:
                stats["preferred_mode"] = preferred_mode
                stats["best_mode"] = self._best_mode_from_variants(stats)
                return stats
        return None

    def _allowed_modes_for_preference(self, preferred: str) -> Optional[set]:
        if preferred == "combined":
            return {"road", "air", "transfer"}
        return {preferred, "transfer"}

    def _filtered_graph(
        self,
        graph: nx.MultiGraph,
        allowed_modes: Optional[set],
        weight: float,
        volume: float,
        created_at: datetime,
    ) -> nx.MultiGraph:
        filtered = nx.MultiGraph()
        for node, data in graph.nodes(data=True):
            if not data.get("is_active", True):
                continue
            if self._node_shutdown(node, data, created_at):
                continue
            filtered.add_node(node, **data)
        for u, v, key, data in graph.edges(keys=True, data=True):
            mode = data.get("mode")
            if allowed_modes is not None and mode not in allowed_modes:
                continue
            if weight > data.get("max_weight_kg", float("inf")):
                continue
            if volume > data.get("max_volume_m3", float("inf")):
                continue
            if not data.get("is_active", True):
                continue
            if self._edge_closed(data, created_at):
                continue
            attrs = dict(data)
            attrs.setdefault("events", list(data.get("events", [])))
            filtered.add_edge(u, v, key=key, **attrs)
        return filtered

    def _evaluate_path(
        self,
        original: nx.MultiGraph,
        filtered: nx.MultiGraph,
        path: List[str],
        weight: float,
        volume: float,
        created_at: datetime,
        weather_index: Dict[Tuple[str, datetime], Dict[str, Any]],
        node_events: Dict[str, List[Dict[str, Any]]],
        mode_events: Dict[str, List[Dict[str, Any]]],
        air_schedule: Dict[str, float],
    ) -> Dict[str, Any]:
        total_time = 0.0
        total_cost = 0.0
        reliability = 1.0
        service_time_h = 0.0
        transfer_time_h = 0.0
        waiting_time_h = 0.0
        co2 = 0.0
        modes: List[str] = []
        nodes_seq: List[str] = [path[0]]
        cost_breakdown = {"fuel": 0.0, "handling": 0.0, "terminal": 0.0, "penalty": 0.0}
        share = {"road": 0.0, "air": 0.0}
        congestion_samples: List[float] = []
        risk_profile = {"weather_delay_prob": 0.0, "scenario_delay_prob": 0.0, "cancel_prob": 0.0}
        current_time = created_at
        prev_mode: Optional[str] = None

        for u, v in zip(path[:-1], path[1:]):
            edge = self._pick_edge(filtered, u, v)
            if edge is None:
                continue
            mode = edge["mode"]
            base_time = float(edge.get("base_time_h", 1.0))
            base_cost = float(edge.get("base_cost", 0.0))
            distance = float(edge.get("distance_km", 0.0))
            reliability_edge = float(edge.get("reliability", 0.95))
            co2_rate = float(edge.get("co2_kg_per_km", 0.0))

            service = self._service_time(original.nodes[u], weight, volume)
            service_time_h += service
            current_time += timedelta(hours=service)

            weather_u = self._weather_impact(weather_index, u, current_time, mode)
            weather_v = self._weather_impact(weather_index, v, current_time, mode)
            weather_time = (weather_u["time_mult"] + weather_v["time_mult"]) / 2
            weather_cost = (weather_u["cost_mult"] + weather_v["cost_mult"]) / 2
            weather_penalty = max(weather_u["reliability_penalty"], weather_v["reliability_penalty"])
            risk_profile["weather_delay_prob"] += weather_penalty

            congestion = self._congestion_multiplier(mode, current_time, weather_u, weather_v)
            congestion_samples.append(congestion)

            scenario_effect = self._scenario_effect(edge, current_time)
            risk_profile["scenario_delay_prob"] += scenario_effect["delay_prob"]
            risk_profile["cancel_prob"] += scenario_effect["cancel_prob"]

            if mode == "air":
                wait = self._air_wait_time(edge, current_time, air_schedule)
                waiting_time_h += wait
                current_time += timedelta(hours=wait)

            if prev_mode and prev_mode != mode:
                transfer = self._transfer_cost_time(original.nodes[u], prev_mode, mode)
                transfer_time_h += transfer["time_h"]
                total_cost += transfer["cost"]
                current_time += timedelta(hours=transfer["time_h"])
                cost_breakdown["terminal"] += transfer["cost"]

            effective_time = base_time * weather_time * congestion * scenario_effect["time_mult"]
            effective_cost = base_cost * weather_cost * scenario_effect["cost_mult"]
            total_time += effective_time
            total_cost += effective_cost
            co2 += distance * co2_rate
            reliability *= clamp(reliability_edge - weather_penalty, 0.1, 0.999)

            comp = self._cost_components(mode, base_cost)
            scale = effective_cost / max(1.0, base_cost)
            cost_breakdown["fuel"] += comp["fuel"] * scale
            cost_breakdown["handling"] += comp["handling"] * scale
            cost_breakdown["terminal"] += comp["terminal"] * scale

            share.setdefault(mode, 0.0)
            share[mode] += effective_time

            modes.append(mode)
            nodes_seq.append(v)
            prev_mode = mode

        total_time += service_time_h + transfer_time_h + waiting_time_h
        share_total = sum(share.values()) or 1.0
        for key in share:
            share[key] = share[key] / share_total
        congestion_factor = float(np.mean(congestion_samples)) if congestion_samples else 1.0
        risk_profile = {k: clamp(v, 0.0, 0.95) for k, v in risk_profile.items()}

        weather_origin = weather_index.get((nodes_seq[0], self._day(current_time)))
        weather_dest = weather_index.get((nodes_seq[-1], self._day(current_time)))
        weather_summary = {
            "origin_event": weather_origin["event"] if weather_origin else "unknown",
            "destination_event": weather_dest["event"] if weather_dest else "unknown",
            "severity": float(
                np.mean(
                    [
                        weather_origin["severity"] if weather_origin else 1.0,
                        weather_dest["severity"] if weather_dest else 1.0,
                    ]
                )
            ),
        }
        return {
            "nodes": nodes_seq,
            "modes": modes,
            "time_h": total_time,
            "cost": total_cost,
            "reliability": reliability,
            "capacity_ok": True,
            "actual_mode": self._actual_mode_from_segments(modes),
            "cost_breakdown": cost_breakdown,
            "service_time_h": service_time_h,
            "transfer_time_h": transfer_time_h,
            "waiting_time_h": waiting_time_h,
            "co2_kg": co2,
            "share": share,
            "congestion_factor": congestion_factor,
            "risk_profile": risk_profile,
            "weather_summary": weather_summary,
        }

    def _pick_edge(self, graph: nx.MultiGraph, u: str, v: str) -> Optional[Dict[str, Any]]:
        data = graph.get_edge_data(u, v, default=None)
        if not data:
            return None
        key, edge_data = min(data.items(), key=lambda kv: kv[1].get("base_time_h", 1.0))
        edge = dict(edge_data)
        edge["edge_key"] = key
        return edge

    def _route_variants(
        self,
        graph: nx.MultiGraph,
        origin: str,
        dest: str,
        cargo_class: str,
        weight: float,
        volume: float,
        created_at: datetime,
        weather_index: Dict[Tuple[str, datetime], Dict[str, Any]],
        node_events: Dict[str, List[Dict[str, Any]]],
        mode_events: Dict[str, List[Dict[str, Any]]],
        air_schedule: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        variants = {
            "road": {"allowed": {"road", "transfer"}},
            "air": {"allowed": {"air", "transfer"}},
            "combined": {"allowed": {"road", "air", "transfer"}},
        }
        results: Dict[str, Dict[str, float]] = {}
        for name, spec in variants.items():
            filtered = self._filtered_graph(graph, spec["allowed"], weight, volume, created_at)
            try:
                path = nx.shortest_path(filtered, origin, dest, weight=lambda u, v, d: d.get("base_time_h", 1.0))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                results[name] = {"time": float("inf"), "cost": float("inf"), "score": float("inf")}
                continue
            stats = self._evaluate_path(
                graph,
                filtered,
                path,
                weight,
                volume,
                created_at,
                weather_index,
                node_events,
                mode_events,
                air_schedule,
            )
            score = self._score(stats["time_h"], stats["cost"], stats["reliability"], stats["capacity_ok"])
            results[name] = {"time": stats["time_h"], "cost": stats["cost"], "score": score}
        return results

    def _best_mode_from_variants(self, stats: Dict[str, Any]) -> str:
        return stats.get("actual_mode", "road")

    def _node_shutdown(self, node_id: str, data: Dict[str, Any], current_time: datetime) -> bool:
        for event in data.get("events", []):
            if event["start"] <= current_time <= event["end"]:
                if event.get("payload", {}).get("is_active") is False:
                    return True
        return False

    def _edge_closed(self, data: Dict[str, Any], current_time: datetime) -> bool:
        for event in data.get("events", []):
            if event["start"] <= current_time <= event["end"]:
                if event.get("payload", {}).get("closed"):
                    return True
        return False

    def _scenario_effect(self, data: Dict[str, Any], current_time: datetime) -> Dict[str, float]:
        result = {"time_mult": 1.0, "cost_mult": 1.0, "delay_prob": 0.0, "cancel_prob": 0.0}
        for event in data.get("events", []):
            if not (event["start"] <= current_time <= event["end"]):
                continue
            payload = event.get("payload", {})
            severity = event.get("severity", 1.0)
            if event["event_type"] == "maintenance":
                result["time_mult"] *= payload.get("time_mult", 1.2)
                result["cost_mult"] *= payload.get("cost_mult", 1.1)
                result["delay_prob"] += 0.08 * severity
            elif event["event_type"] in {"weather_shutdown", "road_block"}:
                result["cancel_prob"] += 0.25 * severity
            elif event["event_type"] == "cost_shock":
                additive = payload.get("cost_additive_per_km", 0.0)
                distance = data.get("distance_km", 0.0)
                result["cost_mult"] *= 1.0 + additive * distance / max(1.0, data.get("base_cost", 1.0))
        return result

    def _node_events_index(self, graph: nx.MultiGraph) -> Dict[str, List[Dict[str, Any]]]:
        return {node: data.get("events", []) for node, data in graph.nodes(data=True)}

    def _mode_events_index(self, graph: nx.MultiGraph) -> Dict[str, List[Dict[str, Any]]]:
        events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for _, _, data in graph.edges(data=True):
            for event in data.get("events", []):
                mode = data.get("mode")
                events[mode].append(event)
        return events

    def _weather_index(self, weather: pd.DataFrame) -> Dict[Tuple[str, datetime], Dict[str, Any]]:
        indexed: Dict[Tuple[str, datetime], Dict[str, Any]] = {}
        for _, row in weather.iterrows():
            indexed[(row["node_id"], row["date"])] = row.to_dict()
        return indexed

    def _weather_impact(
        self,
        weather_index: Dict[Tuple[str, datetime], Dict[str, Any]],
        node_id: str,
        dt: datetime,
        mode: str,
    ) -> Dict[str, float]:
        key = (node_id, self._day(dt))
        row = weather_index.get(key)
        if not row:
            return {"time_mult": 1.0, "cost_mult": 1.0, "reliability_penalty": 0.0}
        mode_key = mode if mode in {"road", "air"} else "transfer"
        return {
            "time_mult": float(row.get(f"time_factor_{mode_key}", 1.0)),
            "cost_mult": float(row.get(f"cost_factor_{mode_key}", 1.0)),
            "reliability_penalty": float(row.get(f"reliability_penalty_{mode_key}", 0.0)),
        }

    def _day(self, dt: datetime) -> datetime:
        return datetime(dt.year, dt.month, dt.day)

    def _congestion_multiplier(
        self,
        mode: str,
        current_time: datetime,
        weather_u: Dict[str, float],
        weather_v: Dict[str, float],
    ) -> float:
        if mode != "road":
            return 1.0
        hour = current_time.hour
        multiplier = 1.0
        for start, end in self.cfg.road_peak_hours:
            if start <= hour < end:
                multiplier *= float(self.rng.uniform(1.12, 1.35))
                break
        if current_time.weekday() >= 5:
            multiplier *= self.cfg.weekend_peak_multiplier
        multiplier *= (weather_u["time_mult"] + weather_v["time_mult"]) / 2
        return multiplier

    def _service_time(self, node_attrs: Dict[str, Any], weight: float, volume: float) -> float:
        base_min = node_attrs.get("service_time_min", 40)
        weight_factor = 0.0004 * weight
        volume_factor = 0.01 * (volume / 12.0)
        return (base_min / 60.0) * (1.0 + weight_factor + volume_factor)

    def _transfer_cost_time(self, node_attrs: Dict[str, Any], from_mode: str, to_mode: str) -> Dict[str, float]:
        profiles = node_attrs.get("transfer_profiles", {})
        key = f"{from_mode}->{to_mode}"
        profile = profiles.get(key) or {"time_h": 0.9, "cost": 980.0}
        return profile

    def _air_wait_time(self, edge_data: Dict[str, Any], current_time: datetime, air_schedule: Dict[str, float]) -> float:
        freq = max(2.0, edge_data.get("schedule_frequency_h", 6.0))
        cutoff = float(edge_data.get("cutoff_hours", 2.0))
        remainder = current_time.hour % freq
        if remainder < cutoff:
            return cutoff - remainder
        return freq - remainder

    def _actual_mode_from_segments(self, modes: List[str]) -> str:
        atomic = {m for m in modes if m in {"road", "air"}}
        if len(atomic) >= 2:
            return "combined"
        if len(atomic) == 1:
            return next(iter(atomic))
        return "road"

    def _cost_components(self, mode: str, base_cost: float) -> Dict[str, float]:
        split = {
            "road": {"fuel": 0.55, "handling": 0.15, "terminal": 0.2, "penalty": 0.1},
            "air": {"fuel": 0.48, "handling": 0.22, "terminal": 0.24, "penalty": 0.06},
            "transfer": {"fuel": 0.1, "handling": 0.4, "terminal": 0.5, "penalty": 0.0},
        }
        shares = split.get(mode, split["road"])
        return {name: base_cost * share for name, share in shares.items()}

    def _build_od_weights(
        self,
        node_ids: List[str],
        type_map: Dict[str, str],
        lat_map: Dict[str, float],
        lon_map: Dict[str, float],
    ) -> List[Tuple[str, str, float]]:
        weights: List[Tuple[str, str, float]] = []
        origin_bias = {"warehouse": 2.2, "hub": 1.6, "airport": 1.2}
        dest_bias = {"warehouse": 1.0, "hub": 1.4, "airport": 2.1}
        for origin in node_ids:
            for dest in node_ids:
                if origin == dest:
                    continue
                dist = max(40.0, haversine(lat_map[origin], lon_map[origin], lat_map[dest], lon_map[dest]))
                weight = (
                    origin_bias.get(type_map[origin], 1.0)
                    * dest_bias.get(type_map[dest], 1.0)
                    / (dist ** self.cfg.od_distance_decay)
                )
                weights.append((origin, dest, float(weight)))
        return weights

    def _od_cumulative(self, weights: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
        cumulative: List[Tuple[str, str, float]] = []
        total = 0.0
        for origin, dest, weight in weights:
            total += weight
            cumulative.append((origin, dest, total))
        if total <= 0:
            return [(weights[0][0], weights[0][1], 1.0)]
        return [(o, d, c / total) for o, d, c in cumulative]

    def _sample_od_pair(self, cumulative: List[Tuple[str, str, float]]) -> Tuple[str, str]:
        r = self.rng.uniform()
        for origin, dest, threshold in cumulative:
            if r <= threshold:
                return origin, dest
        return cumulative[-1][0], cumulative[-1][1]

    def _holiday_info(self, date: datetime) -> Tuple[bool, str]:
        holidays = {
            (1, 1): "new_year",
            (5, 1): "labour_day",
            (7, 4): "independence",
            (12, 25): "xmas",
        }
        key = (date.month, date.day)
        return (key in holidays, holidays.get(key, ""))

    def _weekday_multiplier(self, dow: int) -> float:
        if dow in (0, 1, 2):
            return 1.08
        if dow in (3, 4):
            return 1.12
        if dow == 5:
            return 0.92
        return 0.78

    def _seasonal_multiplier(self, month: int) -> float:
        if month in (11, 12, 1):
            return 1.26
        if month in (2, 3, 4):
            return 1.08
        if month in (5, 6, 7):
            return 0.94
        return 1.02

    def _daily_weather_bias(self, weather: pd.DataFrame, date: datetime) -> float:
        subset = weather[weather["date"] == date]
        if subset.empty:
            return 1.0
        severe = subset["event"].isin(["storm", "snow", "ice"]).mean()
        return clamp(1.1 - severe * 0.45, 0.55, 1.15)

    def _split_for_date(self, dt: datetime) -> str:
        horizon = self._history_anchor()
        if dt >= horizon - timedelta(days=14):
            return "test"
        if dt >= horizon - timedelta(days=28):
            return "val"
        return "train"

    def _cargo_class(self) -> str:
        return self.rng.choice(
            ["standard", "express", "fragile", "refrigerated"],
            p=[0.55, 0.18, 0.17, 0.1],
        )

    def _cargo_size(self, cargo_class: str) -> Tuple[float, float]:
        if cargo_class == "express":
            return float(self.rng.uniform(50, 600)), float(self.rng.uniform(2, 18))
        if cargo_class == "fragile":
            return float(self.rng.uniform(80, 1200)), float(self.rng.uniform(3, 22))
        if cargo_class == "refrigerated":
            return float(self.rng.uniform(500, 3000)), float(self.rng.uniform(10, 42))
        return float(self.rng.uniform(200, 5000)), float(self.rng.uniform(6, 60))

    def _random_time(self, date: datetime) -> datetime:
        seconds = int(self.rng.uniform(0, 24 * 3600))
        return date + timedelta(seconds=seconds)

    def _priority(self, cargo_class: str) -> int:
        mapping = {"express": 3, "fragile": 2, "refrigerated": 2, "standard": 1}
        return mapping.get(cargo_class, 1)

    def _preferred_mode(self, cargo_class: str) -> str:
        if cargo_class == "express":
            return "air"
        if cargo_class == "refrigerated":
            return self.rng.choice(["road", "air"], p=[0.6, 0.4])
        if cargo_class == "fragile":
            return self.rng.choice(["road", "combined"], p=[0.55, 0.45])
        return self.rng.choice(["road", "air", "combined"], p=[0.5, 0.3, 0.2])

    def _score(self, time_h: float, cost: float, reliability: float, capacity_ok: bool) -> float:
        reliable = reliability if capacity_ok else reliability * 0.6
        return (
            self.cfg.w1 * (time_h / 100.0)
            + self.cfg.w2 * (cost / 10000.0)
            + self.cfg.w3 * (1.0 - reliable)
            + self.cfg.w4 * (0.0 if capacity_ok else 1.0)
        )

    def _estimate_time_cost(self, dist_km: float, mode: str, weight: float, volume: float) -> Tuple[float, float]:
        if mode == "road":
            speed = 65.0
            cost_per_km = 44.0
            fixed = 420.0
        elif mode == "air":
            speed = 720.0
            cost_per_km = 180.0
            fixed = 12800.0
        else:
            speed = 70.0
            cost_per_km = 28.0
            fixed = 980.0
        base_time = dist_km / speed
        size_factor = 1.0 + 0.00004 * weight + 0.009 * (volume / 15.0)
        return base_time * size_factor, dist_km * cost_per_km + fixed

    def _config_metadata(self) -> Dict[str, Any]:
        return {
            "osm_mode": self.cfg.osm_mode,
            "bbox": self.cfg.bbox,
            "place_name": self.cfg.place_name,
            "n_nodes": self.cfg.n_nodes,
            "history_days": self.cfg.history_days,
            "mean_orders_per_day": self.cfg.mean_orders_per_day,
            "export_profile": self.cfg.export_profile,
            "od_distance_decay": self.cfg.od_distance_decay,
            "transfer_radius_km": self.cfg.transfer_radius_km,
        }

    def _validate_generation(self, orders: pd.DataFrame, nodes: pd.DataFrame, edges: pd.DataFrame) -> Dict[str, Any]:
        validations: Dict[str, Any] = {}
        if orders.empty:
            return validations
        validations["reachable_od_share"] = float((orders["status"] != "cancelled").mean())
        validations["avg_co2"] = float(orders["co2_kg"].mean())
        validations["mode_distribution"] = orders["actual_mode"].value_counts(normalize=True).to_dict()
        validations["anomaly_share"] = float((orders["status"] != "delivered").mean())
        validations["teleport_check"] = float((orders["actual_time_h"] > 0).mean())
        validations["orders"] = int(len(orders))
        validations["nodes"] = int(len(nodes))
        validations["edges"] = int(len(edges))
        return validations


if __name__ == "__main__":
    cfg = GeneratorConfig(
        osm_mode="bbox",
        bbox=(45.0, 55.0, 60.0, 70.0),
        n_nodes=100,
        history_days=120,
        mean_orders_per_day=150,
        seed=42,
        export_profile="standard",
    )
    generator = DataGenerator(cfg)
    artifacts = generator.build_dataset()
    generator.export(artifacts)
    print(
        "Nodes:", artifacts.nodes.shape,
        "Edges:", artifacts.edges.shape,
        "Orders:", artifacts.orders.shape,
    )
