"""FastAPI service for RailPulse UK cascade forecasting and NRCoT Delay Repay auditing."""

from __future__ import annotations

import math
import os
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agents.delay_repay_copilot import DelayRepayCopilot
from src.analytics.network_cascade import (
    build_uk_rail_graph,
    compute_station_cascade_metrics,
    propagate_knock_on_delay,
)
from src.ingestion.rail_telemetry import UK_STATION_CODES, UK_STATION_METADATA
from src.models.cascade_and_gnn import (
    CORRIDOR_ENCODING,
    DELAY_CODE_ENCODING,
    WEATHER_ENCODING,
)

app = FastAPI(
    title="RailPulse UK API",
    version="0.1.0",
    description="UK Rail Delay Cascade Forecasting, Graph Propagation, and NRCoT Delay Repay Copilot",
)

_GRAPH = build_uk_rail_graph()
_STATION_METRICS = compute_station_cascade_metrics(_GRAPH)
_COPILOT = DelayRepayCopilot()


def _load_conformal_models() -> dict[str, Any]:
    bundle_path = Path(os.environ.get("RAILPULSE_MODEL_BUNDLE", "data/processed/conformal_models.pkl"))
    if bundle_path.exists():
        with bundle_path.open("rb") as f:
            return pickle.load(f)
    return {}


class CascadePredictionRequest(BaseModel):
    """Input payload for downstream cascade delay forecasting."""

    crs_code: str = Field(..., examples=["CRE"])
    corridor: str | None = Field(default=None, examples=["WCML"])
    current_delay_min: float = Field(..., ge=0.0, le=600.0, examples=[28.0])
    turnaround_slack_min: float = Field(default=6.0, ge=0.0, le=180.0)
    weather_regime: str = Field(default="Normal")
    delay_code: str = Field(default="YI")
    is_peak_commuter: bool = Field(default=True)
    departure_hour: int = Field(default=8, ge=0, le=23)
    simulate_feed_outage: bool = Field(
        default=False,
        description="Simulate a Darwin/TRUST live telemetry outage to trigger deterministic fallback.",
    )


class DelayRepayAuditRequest(BaseModel):
    """Input payload for NRCoT / TOC Delay Repay auditing."""

    query: str = Field(..., examples=["Delayed 42 mins on Avanti split ticket calling at Crewe"])
    toc: str = Field(..., examples=["Avanti West Coast"])
    delay_minutes: float = Field(..., ge=0.0, le=1440.0)
    ticket_type: str = Field(default="single", pattern="^(single|return)$")
    is_split_ticket: bool = Field(default=False)
    train_calls_at_split_station: bool = Field(default=True)
    has_zonal_or_season_split: bool = Field(default=False)
    ticket_price_gbp: float | None = Field(default=None, ge=0.0)


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Check API, graph, model artifact, and telemetry feed readiness."""
    models = _load_conformal_models()
    duckdb_exists = Path(os.environ.get("RAILPULSE_DUCKDB_PATH", "data/processed/railpulse.duckdb")).exists()
    live_feed_env = os.environ.get("RAILPULSE_LIVE_FEED_AVAILABLE", "true").lower() == "true"
    return {
        "status": "ok",
        "service": "railpulse-uk",
        "stations_loaded": len(_GRAPH.nodes),
        "conformal_models_loaded": sorted(models.keys()),
        "duckdb_ready": duckdb_exists,
        "feed_status": "live" if (live_feed_env and bool(models)) else "degraded_fallback",
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


def _degraded_fallback_forecast(req: CascadePredictionRequest, corridor: str) -> dict[str, dict[str, float]]:
    """Deterministic fallback predictor used when Darwin/TRUST live feed or models are unavailable."""
    unabsorbed = max(0.0, req.current_delay_min - 0.75 * req.turnaround_slack_min)
    weather_mult = 1.35 if req.weather_regime == "Named Winter Storm" else 1.05
    infra_add = 8.0 if req.delay_code in ("I1", "I2", "OT") else 0.0
    peak_add = 3.0 if req.is_peak_commuter else 0.0

    p30 = round(max(0.0, 0.78 * req.current_delay_min + 0.35 * unabsorbed * weather_mult + 0.3 * infra_add), 2)
    p60 = round(max(0.0, 0.62 * req.current_delay_min + 0.50 * unabsorbed * weather_mult + 0.5 * infra_add + peak_add), 2)
    p120 = round(max(0.0, 0.40 * req.current_delay_min + 0.42 * unabsorbed * weather_mult + 0.4 * infra_add), 2)

    return {
        "t+30m": {"point_min": p30, "lower_90_min": round(max(0.0, p30 - 6.0), 2), "upper_90_min": round(p30 + 9.0, 2)},
        "t+60m": {"point_min": p60, "lower_90_min": round(max(0.0, p60 - 8.0), 2), "upper_90_min": round(p60 + 12.0, 2)},
        "t+120m": {"point_min": p120, "lower_90_min": round(max(0.0, p120 - 9.0), 2), "upper_90_min": round(p120 + 14.0, 2)},
    }


@app.post("/predict-cascade")
def predict_cascade(req: CascadePredictionRequest) -> dict[str, Any]:
    """Predict t+30m/t+60m/t+120m cascade delays with 90% conformal intervals and outage fallback."""
    crs = req.crs_code.strip().upper()
    if crs not in UK_STATION_CODES:
        raise HTTPException(status_code=400, detail=f"Unknown UK station CRS code: {crs}")

    corridor = req.corridor or str(UK_STATION_METADATA[crs]["corridor"])
    live_feed_env = os.environ.get("RAILPULSE_LIVE_FEED_AVAILABLE", "true").lower() == "true"
    models = _load_conformal_models()

    use_fallback = req.simulate_feed_outage or (not live_feed_env) or (len(models) == 0)
    propagation_df = propagate_knock_on_delay(
        origin_crs=crs,
        initial_delay_min=req.current_delay_min,
        turnaround_slack_min=req.turnaround_slack_min,
        graph=_GRAPH,
    )

    if use_fallback:
        forecasts = _degraded_fallback_forecast(req, corridor)
        feed_mode = "degraded_fallback"
        fallback_reason = (
            "Live Darwin/TRUST feed unavailable; serving NetworkX graph + statutory slack fallback."
        )
    else:
        row_meta = _STATION_METRICS[_STATION_METRICS["crs_code"] == crs].iloc[0]
        unabsorbed = max(0.0, req.current_delay_min - 0.85 * req.turnaround_slack_min)
        knock_on_score = unabsorbed * (1.0 + 4.0 * float(row_meta["pagerank_centrality"]))
        x_vec = np.array(
            [
                [
                    float(req.current_delay_min),
                    float(req.turnaround_slack_min),
                    float(row_meta["pagerank_centrality"]),
                    float(row_meta["betweenness_centrality"]),
                    float(row_meta["in_degree"]),
                    float(row_meta["out_degree"]),
                    float(knock_on_score),
                    float(int(req.is_peak_commuter)),
                    float(WEATHER_ENCODING.get(req.weather_regime, 0)),
                    float(DELAY_CODE_ENCODING.get(req.delay_code, 2)),
                    float(CORRIDOR_ENCODING.get(corridor, 0)),
                    math.sin(2.0 * math.pi * req.departure_hour / 24.0),
                    math.cos(2.0 * math.pi * req.departure_hour / 24.0),
                ]
            ],
            dtype=float,
        )

        forecasts = {}
        for horizon, model in models.items():
            pred, pis = model.predict_interval(x_vec)
            pt = round(max(0.0, float(pred[0])), 2)
            lo = round(max(0.0, float(pis[0, 0, 0])), 2)
            hi = round(max(lo, float(pis[0, 1, 0])), 2)
            forecasts[horizon] = {
                "point_min": pt,
                "lower_90_min": lo,
                "upper_90_min": hi,
            }
        feed_mode = "live"
        fallback_reason = None

    return {
        "crs_code": crs,
        "station_name": UK_STATION_METADATA[crs]["name"],
        "corridor": corridor,
        "feed_mode": feed_mode,
        "fallback_used": use_fallback,
        "fallback_reason": fallback_reason,
        "horizon_forecasts": forecasts,
        "downstream_propagation": propagation_df.head(6).to_dict(orient="records"),
    }


@app.post("/audit-delay-repay")
def audit_delay_repay(req: DelayRepayAuditRequest) -> dict[str, Any]:
    """Run the guardrailed LangGraph NRCoT & TOC Delay Repay copilot."""
    return _COPILOT.audit_claim(
        query=req.query,
        toc=req.toc,
        delay_minutes=req.delay_minutes,
        ticket_type=req.ticket_type,
        is_split_ticket=req.is_split_ticket,
        train_calls_at_split_station=req.train_calls_at_split_station,
        has_zonal_or_season_split=req.has_zonal_or_season_split,
        ticket_price_gbp=req.ticket_price_gbp,
    )
