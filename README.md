# RailPulse UK

**UK Rail Delay Cascade Forecasting, Spatio-Temporal Graph Neural Network, and NRCoT/ORR Delay Repay Copilot**

RailPulse UK is an end-to-end telemetry ingestion, network graph analytics, conformal + graph neural forecasting, and statutory passenger compensation auditing platform tailored to Great Britain's National Rail network (Network Rail TRUST/Darwin feeds, Met Office weather regimes, and NRCoT / ORR Delay Repay rules).

## Architecture Overview

1. **Telemetry Ingestion & GDPR Quarantine (`src/ingestion/rail_telemetry.py`)**
   - Pandera domain & temporal-order validation (`DarwinTrustSchema`): enforces valid UK station CRS/TIPLOC codes, temporal monotonicity (`actual_arrival >= actual_departure_prev`), and Network Rail Delay Attribution codes (`ON_TIME`, `TG`, `X8`, `I1`, `M2`, `YI`, etc.).
   - UK GDPR passenger booking PNR pseudonymisation via keyed HMAC-SHA256.
   - DuckDB analytical persistence (`data/processed/railpulse.duckdb`) and Parquet Dead-Letter Quarantine (`data/quarantine/darwin_quarantine.parquet`) for corrupted or out-of-order Darwin messages.

2. **Rail Corridor Network Cascade Analytics (`src/analytics/network_cascade.py`)**
   - Directed `NetworkX` corridor graph across major UK hubs (`LIV`, `MAN`, `CRE`, `BHM`, `EUS`, `YRK`, `EDB`, `PRE`, `GLC`, `LDS`, `KGX`).
   - Computes station PageRank cascade centrality, rolling-stock turnaround slack, and knock-on delay propagation scores.

3. **Conformal LightGBM & Spatio-Temporal GAT (`src/models/cascade_and_gnn.py`)**
   - Route-grouped LightGBM + `MAPIE` 90% conformal regression predicting downstream arrival delay at `t+30m`, `t+60m`, and `t+120m` using `GroupKFold` by rail corridor, logged to MLflow SQLite (`sqlite:///mlruns/mlflow.db`).
   - PyTorch Graph Attention / Spatio-Temporal Propagation Network (`RailCascadeGAT`) operating over the UK station adjacency matrix to capture multi-hop knock-on cascades.

4. **Sliced Error Analysis (`src/evaluation/error_analysis.py`)**
   - Generates `docs/ERROR_ANALYSIS_REPORT.md` auditing MAE, RMSE, and 90% conformal interval coverage across operational slices (`Peak Commuter`, `Named Winter Storm`, `Signal/Infrastructure Failure`, `Normal`), alongside GNN vs. LightGBM tail-delay error comparison.

5. **NRCoT & TOC Delay Repay Copilot (`src/agents/delay_repay_copilot.py`)**
   - Guardrailed `LangGraph` workflow with hybrid BM25 + TF-IDF retrieval over the National Rail Conditions of Travel (NRCoT), TOC Passenger Charters (Avanti West Coast 15-min, Northern 15-min, LNER 30-min, TransPennine Express 15-min), and split-ticket validity rules.
   - Citation-verifier judge node + deterministic statutory Delay Repay percentage fallback calculator.

6. **API & Interactive Dashboard (`src/app/main.py`, `src/app/dashboard.py`)**
   - FastAPI endpoints: `/health`, `/predict-cascade`, and `/audit-delay-repay` with automatic degraded-feed fallback during live Darwin/TRUST outages.
   - Streamlit UI for corridor delay heatmaps and interactive Delay Repay auditing.

## Quickstart

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]" streamlit tabulate

# Run the end-to-end pipeline
python -m src.run_pipeline

# Run tests and linting
pytest -v
ruff check src tests

# Start API & Streamlit UI
uvicorn src.app.main:app --reload --port 8000
streamlit run src/app/dashboard.py
```

## Docker Compose Stack

```bash
docker compose up --build
```
Services: `pgvector` (`5432`), `mlflow` (`5000`), and `api` (`8000`).
