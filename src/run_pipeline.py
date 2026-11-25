"""End-to-end RailPulse UK pipeline: synthetic TRUST/Darwin generation, quarantine, cascade ML/GNN, and reporting."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.delay_repay_copilot import DelayRepayCopilot
from src.analytics.network_cascade import (
    build_uk_rail_graph,
    compute_station_cascade_metrics,
    enrich_telemetry_with_network_features,
)
from src.evaluation.error_analysis import generate_error_analysis_report
from src.ingestion.rail_telemetry import (
    UK_STATION_CODES,
    UK_STATION_METADATA,
    ingest_darwin_trust_telemetry,
)
from src.models.cascade_and_gnn import (
    train_conformal_cascade_models,
    train_rail_cascade_gat,
)


def generate_synthetic_uk_rail_telemetry(
    n_rows: int = 720,
    corrupt_fraction: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic UK TRUST/Darwin + Met Office telemetry with ~3% corrupt/out-of-order rows."""
    rng = np.random.default_rng(seed)
    stations = sorted(UK_STATION_CODES.keys())
    base_start = pd.Timestamp("2026-01-14 06:00:00")

    weather_choices = ["Normal", "Named Winter Storm", "Heavy Rain", "High Wind", "Frost/Ice"]
    weather_probs = [0.58, 0.14, 0.12, 0.10, 0.06]

    rows: list[dict[str, object]] = []
    for i in range(n_rows):
        crs = stations[i % len(stations)]
        tiploc = UK_STATION_CODES[crs]
        corridor = str(UK_STATION_METADATA[crs]["corridor"])

        hour_offset = int(i // len(stations))
        sched_arr = base_start + pd.Timedelta(hours=hour_offset, minutes=int(rng.integers(0, 50)))
        run_leg_min = int(rng.integers(22, 65))
        actual_dep_prev = sched_arr - pd.Timedelta(minutes=run_leg_min)

        hour_of_day = sched_arr.hour
        is_peak = (7 <= hour_of_day <= 9) or (16 <= hour_of_day <= 19)
        weather = str(rng.choice(weather_choices, p=weather_probs))

        if weather == "Named Winter Storm":
            delay_code = str(rng.choice(["X8", "YI", "OT"], p=[0.6, 0.25, 0.15]))
            base_delay = float(rng.gamma(shape=3.2, scale=9.5))
        else:
            delay_code = str(
                rng.choice(
                    ["ON_TIME", "TG", "YI", "M2", "I1", "I2", "OT"],
                    p=[0.46, 0.12, 0.16, 0.07, 0.07, 0.08, 0.04],
                )
            )
            if delay_code == "ON_TIME":
                base_delay = float(max(0.0, rng.normal(1.5, 1.5)))
            elif delay_code in ("I1", "I2", "OT"):
                base_delay = float(rng.gamma(shape=2.8, scale=8.5))
            else:
                base_delay = float(rng.gamma(shape=2.0, scale=5.0))

        if is_peak:
            base_delay += float(rng.uniform(2.0, 7.0))

        delay_minutes = round(max(0.0, base_delay), 2)
        actual_arr = sched_arr + pd.Timedelta(minutes=delay_minutes)
        turnaround_slack = round(float(rng.uniform(3.0, 18.0)), 2)

        # Realistic downstream cascade targets incorporating slack absorption and weather amplification
        unabsorbed = max(0.0, delay_minutes - 0.75 * turnaround_slack)
        weather_mult = 1.35 if weather == "Named Winter Storm" else (1.15 if weather in ("Heavy Rain", "High Wind") else 1.0)
        infra_boost = 10.0 if delay_code in ("I1", "I2", "OT") else 0.0
        peak_boost = 3.5 if is_peak else 0.0

        t30 = max(0.0, round(0.78 * delay_minutes + 0.35 * unabsorbed * weather_mult + 0.3 * infra_boost + rng.normal(0, 2.5), 2))
        t60 = max(0.0, round(0.62 * delay_minutes + 0.52 * unabsorbed * weather_mult + 0.5 * infra_boost + peak_boost + rng.normal(0, 3.2), 2))
        t120 = max(0.0, round(0.40 * delay_minutes + 0.45 * unabsorbed * weather_mult + 0.4 * infra_boost + rng.normal(0, 3.8), 2))

        pnr = f"UK{100000 + i:06d}"
        train_uid = f"1A{10 + (i % 89):02d}"
        rid = f"20260114{1000000 + i}"

        rows.append(
            {
                "train_uid": train_uid,
                "rid": rid,
                "corridor": corridor,
                "crs_code": crs,
                "tiploc": tiploc,
                "scheduled_arrival": sched_arr,
                "actual_departure_prev": actual_dep_prev,
                "actual_arrival": actual_arr,
                "delay_minutes": delay_minutes,
                "delay_code": delay_code,
                "weather_regime": weather,
                "is_peak_commuter": bool(is_peak),
                "turnaround_slack_min": turnaround_slack,
                "booking_pnr": pnr,
                "target_delay_30m": t30,
                "target_delay_60m": t60,
                "target_delay_120m": t120,
            }
        )

    df = pd.DataFrame(rows)

    # Inject ~3% realistic Darwin/TRUST corruptions for Dead-Letter Quarantine validation
    n_corrupt = max(3, int(round(n_rows * corrupt_fraction)))
    corrupt_indices = rng.choice(df.index.to_numpy(), size=n_corrupt, replace=False)

    for k, idx in enumerate(corrupt_indices):
        mode = k % 3
        if mode == 0:
            # Temporal order violation: actual_arrival precedes actual_departure_prev
            df.loc[idx, "actual_arrival"] = pd.Timestamp(df.loc[idx, "actual_departure_prev"]) - pd.Timedelta(minutes=12)
        elif mode == 1:
            # Invalid Network Rail Delay Attribution code
            df.loc[idx, "delay_code"] = "BAD_CODE"
        else:
            # CRS / TIPLOC mismatch or invalid UK station code
            df.loc[idx, "crs_code"] = "ZZZ"

    return df


def run_end_to_end_pipeline(
    n_rows: int = 720,
    base_dir: str | Path = ".",
) -> dict[str, object]:
    """Run the full RailPulse UK pipeline and persist artifacts for API & Streamlit UI."""
    root = Path(base_dir)
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    quarantine_dir = root / "data" / "quarantine"
    docs_dir = root / "docs"
    mlruns_dir = root / "mlruns"

    for d in (raw_dir, processed_dir, quarantine_dir, docs_dir, mlruns_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Generate and store raw Darwin/TRUST feed
    raw_df = generate_synthetic_uk_rail_telemetry(n_rows=n_rows, corrupt_fraction=0.03, seed=42)
    raw_path = raw_dir / "darwin_trust_sample.parquet"
    raw_df.to_parquet(raw_path, index=False)

    # 2. Validate contracts, pseudonymise PNRs, quarantine corrupt rows, persist to DuckDB
    duckdb_path = processed_dir / "railpulse.duckdb"
    quarantine_path = quarantine_dir / "darwin_quarantine.parquet"
    ingestion_res = ingest_darwin_trust_telemetry(
        raw_df,
        duckdb_path=duckdb_path,
        quarantine_path=quarantine_path,
    )

    # 3. NetworkX cascade graph enrichment & station metrics
    graph = build_uk_rail_graph()
    enriched_df = enrich_telemetry_with_network_features(ingestion_res.clean_df, graph=graph)
    station_metrics_df = compute_station_cascade_metrics(graph=graph, telemetry_df=enriched_df)

    enriched_path = processed_dir / "enriched_telemetry.parquet"
    station_metrics_path = processed_dir / "station_metrics.parquet"
    enriched_df.to_parquet(enriched_path, index=False)
    station_metrics_df.to_parquet(station_metrics_path, index=False)

    # 4. Conformal LightGBM + RailCascadeGAT PyTorch training + MLflow SQLite logging
    mlflow_db = mlruns_dir / "mlflow.db"
    conformal_results = train_conformal_cascade_models(enriched_df, mlflow_db_path=mlflow_db)
    gat_result = train_rail_cascade_gat(enriched_df, epochs=80, mlflow_db_path=mlflow_db)

    # Persist conformal regressors for FastAPI inference
    model_bundle_path = processed_dir / "conformal_models.pkl"
    with model_bundle_path.open("wb") as f:
        pickle.dump(
            {horizon: res.model for horizon, res in conformal_results.items()},
            f,
        )

    # 5. Sliced Error Analysis Report
    report_path = docs_dir / "ERROR_ANALYSIS_REPORT.md"
    error_bundle = generate_error_analysis_report(
        telemetry_df=enriched_df,
        conformal_results=conformal_results,
        gat_result=gat_result,
        report_path=report_path,
    )

    # 6. Smoke-check LangGraph Delay Repay Copilot
    copilot = DelayRepayCopilot()
    sample_audit = copilot.audit_claim(
        query="Avanti West Coast train delayed 38 minutes on split ticket calling at Crewe",
        toc="Avanti West Coast",
        delay_minutes=38.0,
        ticket_type="single",
        is_split_ticket=True,
        train_calls_at_split_station=True,
        ticket_price_gbp=84.0,
    )

    return {
        "raw_rows": len(raw_df),
        "valid_rows": ingestion_res.valid_count,
        "quarantined_rows": ingestion_res.quarantine_count,
        "duckdb_path": str(duckdb_path),
        "quarantine_path": str(quarantine_path),
        "report_path": str(error_bundle.report_path),
        "mlflow_db": str(mlflow_db),
        "model_bundle_path": str(model_bundle_path),
        "conformal_t60_mae": round(conformal_results["t+60m"].mae, 3),
        "conformal_t60_coverage": round(conformal_results["t+60m"].coverage_90, 4),
        "gat_t60_mae": round(gat_result.mae_60m, 3),
        "gat_tail_mae": round(gat_result.tail_mae_60m, 3),
        "sample_delay_repay_pct": sample_audit["compensation_pct"],
    }


def main() -> None:
    """CLI entrypoint for `python -m src.run_pipeline`."""
    summary = run_end_to_end_pipeline()
    print("=== RailPulse UK Pipeline Completed ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
