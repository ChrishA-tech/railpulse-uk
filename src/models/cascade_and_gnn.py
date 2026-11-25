"""Conformal LightGBM delay cascade regressors (MAPIE + GroupKFold) and PyTorch RailCascadeGAT."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import lightgbm as lgb
import mlflow
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mapie.regression import CrossConformalRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupKFold

from src.analytics.network_cascade import build_uk_rail_graph
from src.ingestion.rail_telemetry import UK_STATION_CODES

torch.set_num_threads(1)

HORIZONS: Final[dict[str, str]] = {
    "t+30m": "target_delay_30m",
    "t+60m": "target_delay_60m",
    "t+120m": "target_delay_120m",
}

WEATHER_ENCODING: Final[dict[str, int]] = {
    "Normal": 0,
    "Frost/Ice": 1,
    "Heavy Rain": 2,
    "High Wind": 3,
    "Named Winter Storm": 4,
}

DELAY_CODE_ENCODING: Final[dict[str, int]] = {
    "ON_TIME": 0,
    "TG": 1,
    "YI": 2,
    "M2": 3,
    "I1": 4,
    "I2": 5,
    "OT": 6,
    "X8": 7,
    "XD": 8,
}

CORRIDOR_ENCODING: Final[dict[str, int]] = {
    "WCML": 0,
    "ECML": 1,
    "TPE": 2,
    "XC": 3,
}

FEATURE_COLUMNS: Final[list[str]] = [
    "delay_minutes",
    "turnaround_slack_min",
    "pagerank_centrality",
    "betweenness_centrality",
    "in_degree",
    "out_degree",
    "knock_on_propagation_score",
    "is_peak_commuter_int",
    "weather_code_idx",
    "delay_code_idx",
    "corridor_idx",
    "hour_sin",
    "hour_cos",
]


def prepare_cascade_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build deterministic tabular feature matrix X and corridor group labels for GroupKFold."""
    work = df.copy()
    ts = pd.to_datetime(work["scheduled_arrival"])
    hours = ts.dt.hour + ts.dt.minute / 60.0

    work["is_peak_commuter_int"] = work["is_peak_commuter"].astype(int)
    work["weather_code_idx"] = work["weather_regime"].map(WEATHER_ENCODING).fillna(0).astype(int)
    work["delay_code_idx"] = work["delay_code"].map(DELAY_CODE_ENCODING).fillna(0).astype(int)
    work["corridor_idx"] = work["corridor"].map(CORRIDOR_ENCODING).fillna(0).astype(int)
    work["hour_sin"] = np.sin(2.0 * math.pi * hours / 24.0)
    work["hour_cos"] = np.cos(2.0 * math.pi * hours / 24.0)

    for col in (
        "pagerank_centrality",
        "betweenness_centrality",
        "in_degree",
        "out_degree",
        "knock_on_propagation_score",
    ):
        if col not in work.columns:
            work[col] = 0.0

    X = work[FEATURE_COLUMNS].astype(float)
    groups = work["corridor"].astype(str)
    return X, groups


@dataclass
class ConformalHorizonResult:
    """Trained conformal LightGBM bundle and out-of-fold / validation predictions for one horizon."""

    horizon: str
    target_col: str
    model: CrossConformalRegressor
    y_true: np.ndarray
    y_pred: np.ndarray
    y_lower: np.ndarray
    y_upper: np.ndarray
    mae: float
    rmse: float
    coverage_90: float
    mean_width: float


def train_conformal_cascade_models(
    df: pd.DataFrame,
    mlflow_db_path: str | Path = "mlruns/mlflow.db",
    experiment_name: str = "railpulse-uk-cascade",
    confidence_level: float = 0.90,
    random_state: int = 42,
) -> dict[str, ConformalHorizonResult]:
    """Train route-grouped LightGBM + MAPIE 90% conformal regressors for t+30m, t+60m, t+120m."""
    mlflow_db = Path(mlflow_db_path)
    mlflow_db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db.resolve()}")
    mlflow.set_experiment(experiment_name)

    X, groups = prepare_cascade_features(df)
    unique_groups = groups.nunique()
    n_splits = max(2, min(4, unique_groups))

    results: dict[str, ConformalHorizonResult] = {}

    for horizon, target_col in HORIZONS.items():
        if target_col not in df.columns:
            raise KeyError(f"Missing target column '{target_col}' in training DataFrame.")
        y = df[target_col].to_numpy(dtype=float)

        base_estimator = lgb.LGBMRegressor(
            n_estimators=120,
            learning_rate=0.06,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=1,
            verbose=-1,
        )

        cv = GroupKFold(n_splits=n_splits)
        conformal = CrossConformalRegressor(
            estimator=base_estimator,
            confidence_level=confidence_level,
            method="plus",
            cv=cv,
            random_state=random_state,
        )
        conformal.fit_conformalize(X.to_numpy(), y, groups=groups.to_numpy())

        y_pred, y_pis = conformal.predict_interval(X.to_numpy())
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
        y_lower = np.clip(np.asarray(y_pis[:, 0, 0], dtype=float), 0.0, None)
        y_upper = np.maximum(y_lower, np.asarray(y_pis[:, 1, 0], dtype=float))

        mae = float(mean_absolute_error(y, y_pred))
        rmse = float(root_mean_squared_error(y, y_pred))
        coverage = float(np.mean((y >= y_lower) & (y <= y_upper)))
        mean_width = float(np.mean(y_upper - y_lower))

        with mlflow.start_run(run_name=f"lgbm_conformal_{horizon}"):
            mlflow.log_params(
                {
                    "model_family": "LightGBM+MAPIE_CrossConformal",
                    "horizon": horizon,
                    "confidence_level": confidence_level,
                    "cv_strategy": f"GroupKFold({n_splits})_by_corridor",
                    "n_samples": len(df),
                    "n_features": X.shape[1],
                }
            )
            mlflow.log_metrics(
                {
                    "mae": mae,
                    "rmse": rmse,
                    "coverage_90": coverage,
                    "mean_interval_width": mean_width,
                }
            )

        results[horizon] = ConformalHorizonResult(
            horizon=horizon,
            target_col=target_col,
            model=conformal,
            y_true=y,
            y_pred=y_pred,
            y_lower=y_lower,
            y_upper=y_upper,
            mae=mae,
            rmse=rmse,
            coverage_90=coverage,
            mean_width=mean_width,
        )

    return results


# ---------------------------------------------------------------------------
# PyTorch Spatio-Temporal Graph Attention Network (RailCascadeGAT)
# ---------------------------------------------------------------------------


class GraphAttentionLayer(nn.Module):
    """Multi-head Graph Attention layer masked by the UK rail station adjacency matrix."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        alpha: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = max(4, out_features // num_heads)
        proj_dim = self.num_heads * self.head_dim

        self.W = nn.Linear(in_features, proj_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(1, num_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(1, num_heads, self.head_dim))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(proj_dim, out_features)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass: x shape (B, N, in_features), adj shape (N, N) or (B, N, N)."""
        batch_size, num_nodes, _ = x.shape
        h = self.W(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)

        # Compute pairwise additive attention scores (B, H, N, N)
        src_scores = (h * self.attn_src).sum(dim=-1).transpose(1, 2).unsqueeze(-1)  # (B, H, N, 1)
        dst_scores = (h * self.attn_dst).sum(dim=-1).transpose(1, 2).unsqueeze(-2)  # (B, H, 1, N)
        e = self.leaky_relu(src_scores + dst_scores)

        if adj.dim() == 2:
            mask = (adj > 0).unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        else:
            mask = (adj > 0).unsqueeze(1)  # (B, 1, N, N)

        e = e.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(e, dim=-1)
        attn_weights = self.dropout(attn_weights)

        h_trans = h.transpose(1, 2)  # (B, H, N, D)
        agg = torch.matmul(attn_weights, h_trans)  # (B, H, N, D)
        agg = agg.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.num_heads * self.head_dim)
        return self.out_proj(agg)


class RailCascadeGAT(nn.Module):
    """Spatio-Temporal Graph Attention Network predicting t+30m, t+60m, t+120m station delays."""

    def __init__(
        self,
        in_features: int = 8,
        hidden_dim: int = 32,
        num_horizons: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(in_features)
        self.gat1 = GraphAttentionLayer(in_features, hidden_dim, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_horizons),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Predict multi-horizon cascade delays (B, N, 3) from station features (B, N, F)."""
        h0 = self.input_norm(x)
        h1 = F.gelu(self.norm1(self.gat1(h0, adj)))
        h2 = F.gelu(self.norm2(self.gat2(h1, adj)) + h1)
        out = self.head(torch.cat([h2, x], dim=-1))
        return F.softplus(out)


def build_station_adjacency(graph: nx.DiGraph | None = None) -> tuple[list[str], torch.Tensor]:
    """Return sorted UK station CRS list and adjacency tensor (N, N) with self-loops."""
    g = graph or build_uk_rail_graph()
    stations = sorted(UK_STATION_CODES.keys())
    idx_map = {crs: i for i, crs in enumerate(stations)}
    n = len(stations)
    adj = np.eye(n, dtype=np.float32)
    for src, dst, data in g.edges(data=True):
        if src in idx_map and dst in idx_map:
            # Include both forward propagation and upstream feedback coupling
            w = float(data.get("weight", 0.8))
            adj[idx_map[src], idx_map[dst]] = w
            adj[idx_map[dst], idx_map[src]] = max(adj[idx_map[dst], idx_map[src]], 0.5 * w)
    return stations, torch.tensor(adj, dtype=torch.float32)


def build_gnn_snapshot_tensors(
    df: pd.DataFrame,
    stations: list[str],
) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    """Aggregate telemetry into hourly UK station-graph snapshots (B, N, 8) and targets (B, N, 3)."""
    work = df.copy()
    work["time_bucket"] = pd.to_datetime(work["scheduled_arrival"]).dt.floor("1h")
    station_idx = {crs: i for i, crs in enumerate(stations)}
    n_stations = len(stations)

    work["weather_sev"] = work["weather_regime"].map(WEATHER_ENCODING).fillna(0).astype(float)
    work["is_infra"] = work["delay_code"].isin(["I1", "I2", "OT"]).astype(float)
    work["is_storm"] = (work["weather_regime"] == "Named Winter Storm").astype(float)
    work["is_peak"] = work["is_peak_commuter"].astype(float)

    buckets = sorted(work["time_bucket"].unique())
    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []
    meta_rows: list[dict[str, object]] = []

    for b_idx, bucket in enumerate(buckets):
        sub = work[work["time_bucket"] == bucket]
        x_mat = np.zeros((n_stations, 8), dtype=np.float32)
        y_mat = np.zeros((n_stations, 3), dtype=np.float32)

        grouped = sub.groupby("crs_code")
        for crs, grp in grouped:
            if crs not in station_idx:
                continue
            i = station_idx[crs]
            x_mat[i, 0] = float(grp["delay_minutes"].mean())
            x_mat[i, 1] = float(grp["delay_minutes"].max())
            x_mat[i, 2] = float(grp["turnaround_slack_min"].mean())
            x_mat[i, 3] = float(grp.get("pagerank_centrality", pd.Series([0.07])).mean()) * 10.0
            x_mat[i, 4] = float(grp["weather_sev"].max())
            x_mat[i, 5] = float(grp["is_peak"].mean())
            x_mat[i, 6] = float(grp["is_infra"].mean())
            x_mat[i, 7] = float(grp["is_storm"].mean())

            y_mat[i, 0] = float(grp["target_delay_30m"].mean())
            y_mat[i, 1] = float(grp["target_delay_60m"].mean())
            y_mat[i, 2] = float(grp["target_delay_120m"].mean())

            meta_rows.append(
                {
                    "snapshot_idx": b_idx,
                    "station_idx": i,
                    "crs_code": crs,
                    "time_bucket": bucket,
                    "corridor": str(grp["corridor"].iloc[0]),
                    "weather_regime": str(grp["weather_regime"].mode().iloc[0]),
                    "delay_code": str(grp["delay_code"].mode().iloc[0]),
                    "is_peak_commuter": bool(grp["is_peak"].mean() >= 0.5),
                    "observed_delay_min": float(grp["delay_minutes"].mean()),
                    "target_delay_30m": y_mat[i, 0],
                    "target_delay_60m": y_mat[i, 1],
                    "target_delay_120m": y_mat[i, 2],
                }
            )

        X_list.append(x_mat)
        Y_list.append(y_mat)

    X_tensor = torch.tensor(np.stack(X_list, axis=0), dtype=torch.float32)
    Y_tensor = torch.tensor(np.stack(Y_list, axis=0), dtype=torch.float32)
    meta_df = pd.DataFrame(meta_rows)
    return X_tensor, Y_tensor, meta_df


@dataclass
class GATTrainingResult:
    """Trained RailCascadeGAT model and snapshot evaluation predictions."""

    model: RailCascadeGAT
    stations: list[str]
    adj: torch.Tensor
    snapshot_eval_df: pd.DataFrame
    mae_60m: float
    rmse_60m: float
    tail_mae_60m: float


def train_rail_cascade_gat(
    df: pd.DataFrame,
    epochs: int = 80,
    lr: float = 0.015,
    mlflow_db_path: str | Path = "mlruns/mlflow.db",
    experiment_name: str = "railpulse-uk-cascade",
    seed: int = 42,
) -> GATTrainingResult:
    """Train PyTorch RailCascadeGAT over UK station snapshots and log metrics to MLflow."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    stations, adj = build_station_adjacency()
    X_tensor, Y_tensor, meta_df = build_gnn_snapshot_tensors(df, stations)

    model = RailCascadeGAT(in_features=X_tensor.shape[-1], hidden_dim=32, num_horizons=3, num_heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Mask to active station observations in each snapshot
    active_mask = torch.zeros((X_tensor.shape[0], len(stations), 1), dtype=torch.bool)
    for _, row in meta_df.iterrows():
        active_mask[int(row["snapshot_idx"]), int(row["station_idx"]), 0] = True

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(X_tensor, adj)
        diff = torch.abs(preds - Y_tensor)
        # Emphasize tail delays (>= 30 min) so GNN tracks storm/signal cascade spikes
        tail_weight = torch.where(Y_tensor >= 30.0, 1.75, 1.0)
        loss_mat = F.smooth_l1_loss(preds, Y_tensor, reduction="none", beta=2.0) * tail_weight
        loss = loss_mat[active_mask.expand_as(diff)].mean()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds_all = model(X_tensor, adj).cpu().numpy()

    eval_df = meta_df.copy()
    eval_df["gnn_pred_30m"] = [
        float(preds_all[int(r.snapshot_idx), int(r.station_idx), 0]) for r in eval_df.itertuples()
    ]
    eval_df["gnn_pred_60m"] = [
        float(preds_all[int(r.snapshot_idx), int(r.station_idx), 1]) for r in eval_df.itertuples()
    ]
    eval_df["gnn_pred_120m"] = [
        float(preds_all[int(r.snapshot_idx), int(r.station_idx), 2]) for r in eval_df.itertuples()
    ]

    mae_60m = float(mean_absolute_error(eval_df["target_delay_60m"], eval_df["gnn_pred_60m"]))
    rmse_60m = float(root_mean_squared_error(eval_df["target_delay_60m"], eval_df["gnn_pred_60m"]))
    tail_sub = eval_df[eval_df["target_delay_60m"] >= 30.0]
    if tail_sub.empty:
        tail_sub = eval_df.nlargest(max(1, len(eval_df) // 5), "target_delay_60m")
    tail_mae_60m = float(mean_absolute_error(tail_sub["target_delay_60m"], tail_sub["gnn_pred_60m"]))

    mlflow_db = Path(mlflow_db_path)
    mlflow_db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db.resolve()}")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="rail_cascade_gat"):
        mlflow.log_params(
            {
                "model_family": "PyTorch_RailCascadeGAT",
                "num_stations": len(stations),
                "num_snapshots": int(X_tensor.shape[0]),
                "epochs": epochs,
                "num_heads": 4,
            }
        )
        mlflow.log_metrics(
            {
                "gnn_mae_60m": mae_60m,
                "gnn_rmse_60m": rmse_60m,
                "gnn_tail_mae_60m": tail_mae_60m,
            }
        )

    return GATTrainingResult(
        model=model,
        stations=stations,
        adj=adj,
        snapshot_eval_df=eval_df,
        mae_60m=mae_60m,
        rmse_60m=rmse_60m,
        tail_mae_60m=tail_mae_60m,
    )
