"""UK Rail Corridor NetworkX graph, PageRank cascade centrality, and knock-on delay propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import networkx as nx
import pandas as pd

from src.ingestion.rail_telemetry import UK_STATION_METADATA

# Directed UK rail corridor edges: (src, dst, corridor, distance_km, run_min, coupling_weight)
UK_RAIL_EDGES: Final[list[tuple[str, str, str, float, float, float]]] = [
    # West Coast Main Line (WCML)
    ("EUS", "BHM", "WCML", 182.0, 82.0, 0.85),
    ("BHM", "EUS", "WCML", 182.0, 84.0, 0.80),
    ("EUS", "CRE", "WCML", 254.0, 95.0, 0.90),
    ("CRE", "EUS", "WCML", 254.0, 96.0, 0.88),
    ("BHM", "CRE", "WCML", 86.0, 48.0, 0.82),
    ("CRE", "BHM", "WCML", 86.0, 49.0, 0.80),
    ("CRE", "LIV", "WCML", 60.0, 36.0, 0.78),
    ("LIV", "CRE", "WCML", 60.0, 36.0, 0.76),
    ("CRE", "MAN", "WCML", 50.0, 34.0, 0.88),
    ("MAN", "CRE", "WCML", 50.0, 35.0, 0.86),
    ("CRE", "PRE", "WCML", 82.0, 44.0, 0.84),
    ("PRE", "CRE", "WCML", 82.0, 45.0, 0.82),
    ("PRE", "GLC", "WCML", 312.0, 138.0, 0.75),
    ("GLC", "PRE", "WCML", 312.0, 140.0, 0.74),
    # TransPennine Express (TPE)
    ("LIV", "MAN", "TPE", 51.0, 35.0, 0.92),
    ("MAN", "LIV", "TPE", 51.0, 35.0, 0.90),
    ("MAN", "LDS", "TPE", 69.0, 52.0, 0.89),
    ("LDS", "MAN", "TPE", 69.0, 53.0, 0.88),
    ("LDS", "YRK", "TPE", 41.0, 26.0, 0.86),
    ("YRK", "LDS", "TPE", 41.0, 27.0, 0.85),
    ("MAN", "PRE", "TPE", 50.0, 40.0, 0.78),
    ("PRE", "MAN", "TPE", 50.0, 41.0, 0.77),
    # East Coast Main Line (ECML)
    ("KGX", "PBO", "ECML", 123.0, 48.0, 0.88),
    ("PBO", "KGX", "ECML", 123.0, 50.0, 0.86),
    ("PBO", "YRK", "ECML", 180.0, 70.0, 0.90),
    ("YRK", "PBO", "ECML", 180.0, 72.0, 0.89),
    ("YRK", "NCL", "ECML", 130.0, 58.0, 0.87),
    ("NCL", "YRK", "ECML", 130.0, 59.0, 0.86),
    ("NCL", "EDB", "ECML", 200.0, 88.0, 0.84),
    ("EDB", "NCL", "ECML", 200.0, 90.0, 0.83),
    ("GLC", "EDB", "ECML", 75.0, 50.0, 0.75),
    ("EDB", "GLC", "ECML", 75.0, 50.0, 0.75),
    # CrossCountry (XC)
    ("BHM", "SHF", "XC", 125.0, 70.0, 0.84),
    ("SHF", "BHM", "XC", 125.0, 72.0, 0.83),
    ("SHF", "LDS", "XC", 62.0, 42.0, 0.85),
    ("LDS", "SHF", "XC", 62.0, 42.0, 0.84),
    ("SHF", "YRK", "XC", 74.0, 48.0, 0.82),
    ("YRK", "SHF", "XC", 74.0, 49.0, 0.81),
    ("SHF", "MAN", "XC", 66.0, 51.0, 0.80),
    ("MAN", "SHF", "XC", 66.0, 52.0, 0.80),
]


@dataclass
class StationCascadeMetrics:
    """Computed centrality and cascade propagation profile for each UK station."""

    crs_code: str
    station_name: str
    corridor: str
    pagerank_centrality: float
    betweenness_centrality: float
    in_degree: int
    out_degree: int
    cascade_vulnerability_score: float


def build_uk_rail_graph() -> nx.DiGraph:
    """Build the directed NetworkX UK rail corridor graph across major hubs."""
    graph = nx.DiGraph()
    for crs, meta in UK_STATION_METADATA.items():
        graph.add_node(
            crs,
            name=meta["name"],
            tiploc=meta["tiploc"],
            lat=meta["lat"],
            lon=meta["lon"],
            corridor=meta["corridor"],
        )

    for src, dst, corridor, distance_km, run_min, weight in UK_RAIL_EDGES:
        graph.add_edge(
            src,
            dst,
            corridor=corridor,
            distance_km=distance_km,
            scheduled_run_min=run_min,
            weight=weight,
        )
    return graph


def compute_station_cascade_metrics(
    graph: nx.DiGraph | None = None,
    telemetry_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute PageRank cascade centrality, betweenness, and vulnerability scores per station."""
    g = graph or build_uk_rail_graph()
    pagerank = nx.pagerank(g, alpha=0.85, weight="weight")
    betweenness = nx.betweenness_centrality(g, weight="scheduled_run_min", normalized=True)

    mean_delay_by_crs: dict[str, float] = {}
    mean_slack_by_crs: dict[str, float] = {}
    if telemetry_df is not None and not telemetry_df.empty:
        grouped = telemetry_df.groupby("crs_code", as_index=True)
        mean_delay_by_crs = grouped["delay_minutes"].mean().to_dict()
        if "turnaround_slack_min" in telemetry_df.columns:
            mean_slack_by_crs = grouped["turnaround_slack_min"].mean().to_dict()

    rows: list[dict[str, object]] = []
    for crs in sorted(g.nodes):
        node_data = g.nodes[crs]
        pr = float(pagerank[crs])
        bw = float(betweenness[crs])
        mean_delay = float(mean_delay_by_crs.get(crs, 6.0))
        mean_slack = float(mean_slack_by_crs.get(crs, 8.0))
        slack_buffer_factor = max(0.25, 1.0 - min(mean_slack, 30.0) / 40.0)
        vulnerability = round(pr * 100.0 * (1.0 + bw) * slack_buffer_factor + 0.15 * max(0.0, mean_delay), 4)

        rows.append(
            {
                "crs_code": crs,
                "station_name": node_data["name"],
                "corridor": node_data["corridor"],
                "lat": float(node_data["lat"]),
                "lon": float(node_data["lon"]),
                "pagerank_centrality": round(pr, 6),
                "betweenness_centrality": round(bw, 6),
                "in_degree": int(g.in_degree(crs)),
                "out_degree": int(g.out_degree(crs)),
                "mean_delay_min": round(mean_delay, 2),
                "mean_turnaround_slack_min": round(mean_slack, 2),
                "cascade_vulnerability_score": vulnerability,
            }
        )

    return pd.DataFrame(rows).sort_values("pagerank_centrality", ascending=False).reset_index(drop=True)


def compute_turnaround_slack_absorption(
    incoming_delay_min: float,
    turnaround_slack_min: float,
    recovery_efficiency: float = 0.85,
) -> dict[str, float]:
    """Compute how much incoming delay is absorbed by rolling-stock turnaround slack."""
    effective_slack = max(0.0, turnaround_slack_min) * recovery_efficiency
    unabsorbed_delay = max(0.0, incoming_delay_min - effective_slack)
    absorbed_delay = max(0.0, incoming_delay_min - unabsorbed_delay)
    return {
        "incoming_delay_min": round(float(incoming_delay_min), 2),
        "turnaround_slack_min": round(float(turnaround_slack_min), 2),
        "absorbed_delay_min": round(float(absorbed_delay), 2),
        "unabsorbed_knock_on_min": round(float(unabsorbed_delay), 2),
    }


def propagate_knock_on_delay(
    origin_crs: str,
    initial_delay_min: float,
    turnaround_slack_min: float = 6.0,
    graph: nx.DiGraph | None = None,
    damping: float = 0.78,
    max_hops: int = 3,
) -> pd.DataFrame:
    """Propagate knock-on delay across downstream UK stations using graph coupling and turnaround slack."""
    g = graph or build_uk_rail_graph()
    if origin_crs not in g:
        raise ValueError(f"Unknown UK station CRS code: {origin_crs}")

    pagerank = nx.pagerank(g, alpha=0.85, weight="weight")
    slack_stats = compute_turnaround_slack_absorption(initial_delay_min, turnaround_slack_min)
    seed_delay = slack_stats["unabsorbed_knock_on_min"]

    propagated: dict[str, tuple[int, float, str]] = {
        origin_crs: (0, float(initial_delay_min), origin_crs)
    }

    frontier: list[tuple[str, int, float, str]] = [(origin_crs, 0, seed_delay, origin_crs)]
    while frontier:
        curr, hop, curr_delay, path_str = frontier.pop(0)
        if hop >= max_hops or curr_delay <= 0.5:
            continue

        for nxt in g.successors(curr):
            edge_weight = float(g[curr][nxt]["weight"])
            centrality_boost = 1.0 + 1.5 * float(pagerank[nxt])
            # Downstream stations retain part of turnaround recovery per hop
            next_delay = max(
                0.0,
                curr_delay * damping * edge_weight * centrality_boost - 0.35 * turnaround_slack_min,
            )
            prev_best = propagated.get(nxt)
            if prev_best is None or next_delay > prev_best[1]:
                new_path = f"{path_str}->{nxt}"
                propagated[nxt] = (hop + 1, next_delay, new_path)
                frontier.append((nxt, hop + 1, next_delay, new_path))

    records = []
    for crs, (hop, delay_val, path_str) in propagated.items():
        records.append(
            {
                "crs_code": crs,
                "station_name": g.nodes[crs]["name"],
                "hop_distance": hop,
                "projected_knock_on_delay_min": round(delay_val, 2),
                "pagerank_centrality": round(float(pagerank[crs]), 6),
                "propagation_path": path_str,
            }
        )

    return pd.DataFrame(records).sort_values(
        ["hop_distance", "projected_knock_on_delay_min"], ascending=[True, False]
    ).reset_index(drop=True)


def enrich_telemetry_with_network_features(
    df: pd.DataFrame,
    graph: nx.DiGraph | None = None,
) -> pd.DataFrame:
    """Attach NetworkX station PageRank, degree, and knock-on risk score to telemetry rows."""
    g = graph or build_uk_rail_graph()
    metrics_df = compute_station_cascade_metrics(g, telemetry_df=df)
    lookup = metrics_df.set_index("crs_code")[
        ["pagerank_centrality", "betweenness_centrality", "in_degree", "out_degree", "cascade_vulnerability_score"]
    ]
    enriched = df.copy()
    for col in lookup.columns:
        enriched[col] = enriched["crs_code"].map(lookup[col]).fillna(0.0)

    # Row-level knock-on propagation score combining unabsorbed turnaround delay and PageRank
    unabsorbed = (enriched["delay_minutes"] - 0.85 * enriched["turnaround_slack_min"]).clip(lower=0.0)
    enriched["knock_on_propagation_score"] = (
        unabsorbed * (1.0 + 4.0 * enriched["pagerank_centrality"])
    ).round(4)
    return enriched
