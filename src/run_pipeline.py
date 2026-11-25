"""End-to-end RailPulse UK pipeline: live UK rail/weather ingestion, quarantine, cascade ML/GNN, and reporting."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import duckdb

from src.agents.delay_repay_copilot import DelayRepayCopilot
from src.analytics.network_cascade import (
    build_uk_rail_graph,
    compute_station_cascade_metrics,
    enrich_telemetry_with_network_features,
)
from src.evaluation.error_analysis import generate_error_analysis_report
from src.ingestion.live_rail_connectors import fetch_live_uk_rail_corridor_telemetry
from src.ingestion.rail_telemetry import (
    generate_synthetic_uk_rail_telemetry,
    ingest_darwin_trust_telemetry,
)
from src.models.cascade_and_gnn import (
    train_conformal_cascade_models,
    train_rail_cascade_gat,
)

__all__ = ["generate_synthetic_uk_rail_telemetry", "run_end_to_end_pipeline", "main"]


def run_end_to_end_pipeline(
    n_rows: int = 720,
    base_dir: str | Path = ".",
    live: bool = True,
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

    duckdb_path = processed_dir / "railpulse.duckdb"
    quarantine_path = quarantine_dir / "darwin_quarantine.parquet"

    # 1 & 2. Ingest live UK rail/weather telemetry (with graceful fallback) or offline synthetic data
    if live:
        clean_df = fetch_live_uk_rail_corridor_telemetry(
            n_rows=n_rows,
            corrupt_fraction=0.03,
            quarantine_path=quarantine_path,
        )
        feed_status = str(clean_df.attrs.get("feed_status", "LIVE"))
        raw_count = int(clean_df.attrs.get("raw_count", len(clean_df)))
        valid_count = int(clean_df.attrs.get("valid_count", len(clean_df)))
        quarantine_count = int(clean_df.attrs.get("quarantine_count", 0))

        # Persist validated clean rows to DuckDB and raw snapshot to data/raw
        clean_df.to_parquet(raw_dir / "darwin_trust_sample.parquet", index=False)
        with duckdb.connect(str(duckdb_path)) as con:
            con.register("clean_telemetry_view", clean_df)
            con.execute(
                "CREATE OR REPLACE TABLE telemetry_events AS SELECT * FROM clean_telemetry_view"
            )
    else:
        raw_df = generate_synthetic_uk_rail_telemetry(n_rows=n_rows, corrupt_fraction=0.03, seed=42)
        raw_df.to_parquet(raw_dir / "darwin_trust_sample.parquet", index=False)
        ingestion_res = ingest_darwin_trust_telemetry(
            raw_df,
            duckdb_path=duckdb_path,
            quarantine_path=quarantine_path,
        )
        clean_df = ingestion_res.clean_df
        feed_status = "OFFLINE_SYNTHETIC"
        raw_count = len(raw_df)
        valid_count = ingestion_res.valid_count
        quarantine_count = ingestion_res.quarantine_count

    # 3. NetworkX cascade graph enrichment & station metrics
    graph = build_uk_rail_graph()
    enriched_df = enrich_telemetry_with_network_features(clean_df, graph=graph)
    station_metrics_df = compute_station_cascade_metrics(graph=graph, telemetry_df=enriched_df)

    enriched_df.to_parquet(processed_dir / "enriched_telemetry.parquet", index=False)
    station_metrics_df.to_parquet(processed_dir / "station_metrics.parquet", index=False)

    # 4. Conformal LightGBM + RailCascadeGAT PyTorch training + MLflow SQLite logging
    mlflow_db = mlruns_dir / "mlflow.db"
    conformal_results = train_conformal_cascade_models(enriched_df, mlflow_db_path=mlflow_db)
    gat_result = train_rail_cascade_gat(enriched_df, epochs=80, mlflow_db_path=mlflow_db)

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

    # 6. Smoke-check LangGraph Delay Repay Copilot backed by data/regulatory_docs/
    copilot = DelayRepayCopilot(docs_dir=root / "data" / "regulatory_docs")
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
        "feed_status": feed_status,
        "raw_rows": raw_count,
        "valid_rows": valid_count,
        "quarantined_rows": quarantine_count,
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
        "regulatory_docs_indexed": len(copilot.retriever.corpus),
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for `python -m src.run_pipeline`."""
    parser = argparse.ArgumentParser(description="Run the RailPulse UK end-to-end pipeline.")
    parser.add_argument(
        "--live",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch live Open-Meteo UK hub weather & TfL/Darwin status (default: --live; use --no-live for offline).",
    )
    parser.add_argument("--rows", type=int, default=720, help="Number of corridor movement rows.")
    args = parser.parse_args(argv)

    summary = run_end_to_end_pipeline(n_rows=args.rows, live=args.live)
    print("=== RailPulse UK Pipeline Completed ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
