# RailPulse UK — 300-Capability Technical Skills Matrix

This matrix maps 300 concrete engineering, data, ML, graph neural network, agentic RAG, and MLOps capabilities directly to implemented files and symbols in `railpulse-uk`.

---

## 1. Rail Telemetry Streaming & Data Contracts (Skills 001–050)

| # | Capability | Implementation File | Function / Symbol |
|---:|---|---|---|
| 001 | Live UK rail corridor telemetry orchestration | `src/ingestion/live_rail_connectors.py` | `fetch_live_uk_rail_corridor_telemetry` |
| 002 | Open-Meteo multi-hub weather API client | `src/ingestion/live_rail_connectors.py` | `_fetch_open_meteo_uk_hubs` |
| 003 | Batched UK hub latitude/longitude parameterisation | `src/ingestion/live_rail_connectors.py` | `_fetch_open_meteo_uk_hubs` |
| 004 | Live wind gust speed extraction (`wind_gusts_10m`) | `src/ingestion/live_rail_connectors.py` | `_fetch_open_meteo_uk_hubs` |
| 005 | Live precipitation intensity extraction | `src/ingestion/live_rail_connectors.py` | `_fetch_open_meteo_uk_hubs` |
| 006 | Freezing railhead temperature detection (`<= 1.0°C`) | `src/ingestion/live_rail_connectors.py` | `classify_uk_weather_regime` |
| 007 | Named Winter Storm wind gust threshold (`>= 75 km/h`) | `src/ingestion/live_rail_connectors.py` | `classify_uk_weather_regime` |
| 008 | High Wind speed restriction regime (`>= 50 km/h`) | `src/ingestion/live_rail_connectors.py` | `classify_uk_weather_regime` |
| 009 | Heavy Rain adhesion regime classification (`>= 3.0 mm`) | `src/ingestion/live_rail_connectors.py` | `classify_uk_weather_regime` |
| 010 | Live TfL Unified API open status ingestion | `src/ingestion/live_rail_connectors.py` | `_fetch_uk_rail_status` |
| 011 | Authenticated National Rail Darwin/Huxley2 board query | `src/ingestion/live_rail_connectors.py` | `_fetch_uk_rail_status` |
| 012 | `DARWIN_TOKEN` environment credential gating | `src/ingestion/live_rail_connectors.py` | `fetch_live_uk_rail_corridor_telemetry` |
| 013 | Unauthenticated calibrated corridor telemetry pairing | `src/ingestion/live_rail_connectors.py` | `fetch_live_uk_rail_corridor_telemetry` |
| 014 | Tenacity exponential backoff HTTP retry policy | `src/ingestion/live_rail_connectors.py` | `_fetch_open_meteo_uk_hubs` |
| 015 | Tenacity rail status retry handling (`httpx.HTTPError`) | `src/ingestion/live_rail_connectors.py` | `_fetch_uk_rail_status` |
| 016 | Graceful `LIVE_RAIL_FEED_DEGRADED` offline fallback | `src/ingestion/live_rail_connectors.py` | `fetch_live_uk_rail_corridor_telemetry` |
| 017 | Injected `httpx.Client` lifecycle & testability | `src/ingestion/live_rail_connectors.py` | `fetch_live_uk_rail_corridor_telemetry` |
| 018 | Core 7 UK rail hub definition (`LIV`..`EDB`) | `src/ingestion/live_rail_connectors.py` | `CORE_LIVE_HUBS` |
| 019 | Live feed validation & quarantine wrapper | `src/ingestion/live_rail_connectors.py` | `_validate_and_quarantine` |
| 020 | Feed provenance metadata (`attrs["feed_status"]`) | `src/ingestion/live_rail_connectors.py` | `fetch_live_uk_rail_corridor_telemetry` |
| 021 | Canonical UK CRS-to-TIPLOC station mapping (14 hubs) | `src/ingestion/rail_telemetry.py` | `UK_STATION_CODES` |
| 022 | UK station geospatial coordinates & corridor map | `src/ingestion/rail_telemetry.py` | `UK_STATION_METADATA` |
| 023 | Network Rail Delay Attribution code registry | `src/ingestion/rail_telemetry.py` | `VALID_DELAY_ATTRIBUTION_CODES` |
| 024 | Valid UK rail corridor registry (`WCML`/`ECML`/`TPE`/`XC`) | `src/ingestion/rail_telemetry.py` | `VALID_CORRIDORS` |
| 025 | Operational weather regime domain enum | `src/ingestion/rail_telemetry.py` | `VALID_WEATHER_REGIMES` |
| 026 | UK GDPR passenger PNR HMAC-SHA256 pseudonymisation | `src/ingestion/rail_telemetry.py` | `pnr_hmac_pseudonymise` |
| 027 | Normalised case-insensitive PNR canonicalisation | `src/ingestion/rail_telemetry.py` | `pnr_hmac_pseudonymise` |
| 028 | Environment-configurable HMAC secret key management | `src/ingestion/rail_telemetry.py` | `DEFAULT_PNR_HMAC_KEY` |
| 029 | Raw `booking_pnr` PII column stripping pre-storage | `src/ingestion/rail_telemetry.py` | `_prepare_telemetry_frame` |
| 030 | Pandera `DarwinTrustSchema` contract definition | `src/ingestion/rail_telemetry.py` | `DarwinTrustSchema` |
| 031 | Train UID regex contract (`^[A-Z0-9]{4,10}$`) | `src/ingestion/rail_telemetry.py` | `DarwinTrustSchema` |
| 032 | Darwin RTTI train ID (`rid`) length contract | `src/ingestion/rail_telemetry.py` | `DarwinTrustSchema` |
| 033 | Delay minutes physical bounds (`-15.0` to `600.0`) | `src/ingestion/rail_telemetry.py` | `DarwinTrustSchema` |
| 034 | Turnaround slack physical bounds (`0.0` to `180.0`) | `src/ingestion/rail_telemetry.py` | `DarwinTrustSchema` |
| 035 | SHA-256 64-hex `pnr_hash` format enforcement | `src/ingestion/rail_telemetry.py` | `DarwinTrustSchema` |
| 036 | Temporal monotonicity check (`actual_arrival >= actual_departure_prev`) | `src/ingestion/rail_telemetry.py` | `_check_temporal_monotonicity` |
| 037 | Cross-field CRS-TIPLOC consistency validation | `src/ingestion/rail_telemetry.py` | `_check_crs_tiploc_match` |
| 038 | Fault-tolerant timestamp coercion (`errors="coerce"`) | `src/ingestion/rail_telemetry.py` | `_prepare_telemetry_frame` |
| 039 | Null/NaT timestamp pre-quarantine filtering | `src/ingestion/rail_telemetry.py` | `ingest_darwin_trust_telemetry` |
| 040 | Pandera lazy validation error aggregation | `src/ingestion/rail_telemetry.py` | `_extract_failure_reasons` |
| 041 | Human-readable `quarantine_reason` annotation | `src/ingestion/rail_telemetry.py` | `ingest_darwin_trust_telemetry` |
| 042 | Parquet Dead-Letter Quarantine writer | `src/ingestion/rail_telemetry.py` | `write_quarantine_parquet` |
| 043 | ISO-safe string timestamp serialisation for quarantine | `src/ingestion/rail_telemetry.py` | `write_quarantine_parquet` |
| 044 | Final strict validation pass on clean records | `src/ingestion/rail_telemetry.py` | `ingest_darwin_trust_telemetry` |
| 045 | DuckDB analytical table persistence (`telemetry_events`) | `src/ingestion/rail_telemetry.py` | `ingest_darwin_trust_telemetry` |
| 046 | Structured `IngestionResult` audit dataclass | `src/ingestion/rail_telemetry.py` | `IngestionResult` |
| 047 | Realistic TRUST/Darwin synthetic corridor generator | `src/ingestion/rail_telemetry.py` | `generate_synthetic_uk_rail_telemetry` |
| 048 | Peak commuter window modelling (`07-09`, `16-19`) | `src/ingestion/rail_telemetry.py` | `generate_synthetic_uk_rail_telemetry` |
| 049 | Controlled ~3% corrupt/out-of-order row injection | `src/ingestion/rail_telemetry.py` | `generate_synthetic_uk_rail_telemetry` |
| 050 | Weather-override & status-boost calibration hooks | `src/ingestion/rail_telemetry.py` | `generate_synthetic_uk_rail_telemetry` |

---

## 2. NetworkX Corridor Graph Analytics (Skills 051–100)

| # | Capability | Implementation File | Function / Symbol |
|---:|---|---|---|
| 051 | Directed UK rail corridor graph construction | `src/analytics/network_cascade.py` | `build_uk_rail_graph` |
| 052 | West Coast Main Line (`WCML`) edge topology | `src/analytics/network_cascade.py` | `UK_RAIL_EDGES` |
| 053 | East Coast Main Line (`ECML`) edge topology | `src/analytics/network_cascade.py` | `UK_RAIL_EDGES` |
| 054 | TransPennine Express (`TPE`) edge topology | `src/analytics/network_cascade.py` | `UK_RAIL_EDGES` |
| 055 | CrossCountry (`XC`) corridor edge topology | `src/analytics/network_cascade.py` | `UK_RAIL_EDGES` |
| 056 | Inter-hub distance (`distance_km`) edge metadata | `src/analytics/network_cascade.py` | `UK_RAIL_EDGES` |
| 057 | Scheduled runtime (`scheduled_run_min`) edge weights | `src/analytics/network_cascade.py` | `UK_RAIL_EDGES` |
| 058 | Directional coupling strength (`weight`) modelling | `src/analytics/network_cascade.py` | `UK_RAIL_EDGES` |
| 059 | Station node metadata attachment (`name`, `tiploc`, `lat`, `lon`) | `src/analytics/network_cascade.py` | `build_uk_rail_graph` |
| 060 | Liverpool Lime Street (`LIV`) hub modelling | `src/analytics/network_cascade.py` | `build_uk_rail_graph` |
| 061 | Manchester Piccadilly (`MAN`) hub modelling | `src/analytics/network_cascade.py` | `build_uk_rail_graph` |
| 062 | Crewe (`CRE`) major WCML interchange modelling | `src/analytics/network_cascade.py` | `build_uk_rail_graph` |
| 063 | Birmingham New Street (`BHM`) hub modelling | `src/analytics/network_cascade.py` | `build_uk_rail_graph` |
| 064 | London Euston (`EUS`) terminus modelling | `src/analytics/network_cascade.py` | `build_uk_rail_graph` |
| 065 | York (`YRK`) ECML/TPE/XC hub modelling | `src/analytics/network_cascade.py` | `build_uk_rail_graph` |
| 066 | Edinburgh Waverley (`EDB`) northern terminus modelling | `src/analytics/network_cascade.py` | `build_uk_rail_graph` |
| 067 | Weighted station PageRank cascade centrality | `src/analytics/network_cascade.py` | `compute_station_cascade_metrics` |
| 068 | Shortest-runtime weighted betweenness centrality | `src/analytics/network_cascade.py` | `compute_station_cascade_metrics` |
| 069 | Station inbound degree (`in_degree`) calculation | `src/analytics/network_cascade.py` | `compute_station_cascade_metrics` |
| 070 | Station outbound degree (`out_degree`) calculation | `src/analytics/network_cascade.py` | `compute_station_cascade_metrics` |
| 071 | Telemetry-backed mean delay aggregation per CRS | `src/analytics/network_cascade.py` | `compute_station_cascade_metrics` |
| 072 | Telemetry-backed mean turnaround slack per CRS | `src/analytics/network_cascade.py` | `compute_station_cascade_metrics` |
| 073 | Slack-adjusted station cascade vulnerability score | `src/analytics/network_cascade.py` | `compute_station_cascade_metrics` |
| 074 | `StationCascadeMetrics` schema dataclass | `src/analytics/network_cascade.py` | `StationCascadeMetrics` |
| 075 | PageRank ranking & sorted station leaderboard | `src/analytics/network_cascade.py` | `compute_station_cascade_metrics` |
| 076 | Rolling-stock turnaround slack absorption model | `src/analytics/network_cascade.py` | `compute_turnaround_slack_absorption` |
| 077 | Turnaround recovery efficiency factor (`0.85`) | `src/analytics/network_cascade.py` | `compute_turnaround_slack_absorption` |
| 078 | Unabsorbed knock-on delay residual computation | `src/analytics/network_cascade.py` | `compute_turnaround_slack_absorption` |
| 079 | Absorbed delay minutes quantification | `src/analytics/network_cascade.py` | `compute_turnaround_slack_absorption` |
| 080 | Non-negative clipping on turnaround slack inputs | `src/analytics/network_cascade.py` | `compute_turnaround_slack_absorption` |
| 081 | Multi-hop BFS knock-on delay propagation engine | `src/analytics/network_cascade.py` | `propagate_knock_on_delay` |
| 082 | Unknown CRS guard with explicit `ValueError` | `src/analytics/network_cascade.py` | `propagate_knock_on_delay` |
| 083 | Hop-distance bounded cascade horizon (`max_hops`) | `src/analytics/network_cascade.py` | `propagate_knock_on_delay` |
| 084 | Edge damping decay (`damping=0.78`) per hop | `src/analytics/network_cascade.py` | `propagate_knock_on_delay` |
| 085 | Centrality-amplified downstream delay boost | `src/analytics/network_cascade.py` | `propagate_knock_on_delay` |
| 086 | Downstream turnaround buffer deduction per hop | `src/analytics/network_cascade.py` | `propagate_knock_on_delay` |
| 087 | Worst-case max-delay path retention per station | `src/analytics/network_cascade.py` | `propagate_knock_on_delay` |
| 088 | Human-readable `propagation_path` trace (`CRE->MAN->LDS`) | `src/analytics/network_cascade.py` | `propagate_knock_on_delay` |
| 089 | Sub-threshold (`<= 0.5 min`) propagation pruning | `src/analytics/network_cascade.py` | `propagate_knock_on_delay` |
| 090 | Sorted propagation table by hop & delay severity | `src/analytics/network_cascade.py` | `propagate_knock_on_delay` |
| 091 | Telemetry-to-graph feature join enrichment | `src/analytics/network_cascade.py` | `enrich_telemetry_with_network_features` |
| 092 | Row-level `pagerank_centrality` feature mapping | `src/analytics/network_cascade.py` | `enrich_telemetry_with_network_features` |
| 093 | Row-level `betweenness_centrality` feature mapping | `src/analytics/network_cascade.py` | `enrich_telemetry_with_network_features` |
| 094 | Row-level `in_degree` & `out_degree` feature mapping | `src/analytics/network_cascade.py` | `enrich_telemetry_with_network_features` |
| 095 | Row-level `cascade_vulnerability_score` mapping | `src/analytics/network_cascade.py` | `enrich_telemetry_with_network_features` |
| 096 | Row-level `knock_on_propagation_score` synthesis | `src/analytics/network_cascade.py` | `enrich_telemetry_with_network_features` |
| 097 | Bidirectional corridor asymmetry modelling | `src/analytics/network_cascade.py` | `UK_RAIL_EDGES` |
| 098 | Cross-corridor Scottish interconnection (`GLC<->EDB`) | `src/analytics/network_cascade.py` | `UK_RAIL_EDGES` |
| 099 | TransPennine-to-WCML junction coupling (`MAN<->PRE`) | `src/analytics/network_cascade.py` | `UK_RAIL_EDGES` |
| 100 | CrossCountry-to-ECML junction coupling (`SHF<->YRK`) | `src/analytics/network_cascade.py` | `UK_RAIL_EDGES` |

---

## 3. Conformal Delay Forecasting & Error Auditing (Skills 101–150)

| # | Capability | Implementation File | Function / Symbol |
|---:|---|---|---|
| 101 | Multi-horizon forecast target definitions (`t+30m`/`60m`/`120m`) | `src/models/cascade_and_gnn.py` | `HORIZONS` |
| 102 | Deterministic weather regime integer encoding | `src/models/cascade_and_gnn.py` | `WEATHER_ENCODING` |
| 103 | Deterministic delay attribution code encoding | `src/models/cascade_and_gnn.py` | `DELAY_CODE_ENCODING` |
| 104 | Deterministic rail corridor integer encoding | `src/models/cascade_and_gnn.py` | `CORRIDOR_ENCODING` |
| 105 | Canonical 13-feature tabular contract | `src/models/cascade_and_gnn.py` | `FEATURE_COLUMNS` |
| 106 | Cyclical hour-of-day sine encoding (`hour_sin`) | `src/models/cascade_and_gnn.py` | `prepare_cascade_features` |
| 107 | Cyclical hour-of-day cosine encoding (`hour_cos`) | `src/models/cascade_and_gnn.py` | `prepare_cascade_features` |
| 108 | Peak commuter binary feature extraction | `src/models/cascade_and_gnn.py` | `prepare_cascade_features` |
| 109 | Defensive default fill for optional graph columns | `src/models/cascade_and_gnn.py` | `prepare_cascade_features` |
| 110 | Corridor group label extraction for cross-validation | `src/models/cascade_and_gnn.py` | `prepare_cascade_features` |
| 111 | Route-grouped `GroupKFold` cross-validation | `src/models/cascade_and_gnn.py` | `train_conformal_cascade_models` |
| 112 | Dynamic fold count bounding (`min(4, unique_groups)`) | `src/models/cascade_and_gnn.py` | `train_conformal_cascade_models` |
| 113 | LightGBM gradient boosted regressor configuration | `src/models/cascade_and_gnn.py` | `train_conformal_cascade_models` |
| 114 | MAPIE `CrossConformalRegressor` (`method="plus"`) | `src/models/cascade_and_gnn.py` | `train_conformal_cascade_models` |
| 115 | 90% nominal conformal prediction interval calibration | `src/models/cascade_and_gnn.py` | `train_conformal_cascade_models` |
| 116 | Non-negative delay lower-bound clipping | `src/models/cascade_and_gnn.py` | `train_conformal_cascade_models` |
| 117 | Monotonic interval ordering (`y_upper >= y_lower`) | `src/models/cascade_and_gnn.py` | `train_conformal_cascade_models` |
| 118 | Target column presence assertion guard | `src/models/cascade_and_gnn.py` | `train_conformal_cascade_models` |
| 119 | Horizon MAE evaluation (`mean_absolute_error`) | `src/models/cascade_and_gnn.py` | `train_conformal_cascade_models` |
| 120 | Horizon RMSE evaluation (`root_mean_squared_error`) | `src/models/cascade_and_gnn.py` | `train_conformal_cascade_models` |
| 121 | Empirical 90% conformal coverage calculation | `src/models/cascade_and_gnn.py` | `train_conformal_cascade_models` |
| 122 | Mean conformal interval width sharpness metric | `src/models/cascade_and_gnn.py` | `train_conformal_cascade_models` |
| 123 | `ConformalHorizonResult` structured result container | `src/models/cascade_and_gnn.py` | `ConformalHorizonResult` |
| 124 | Multi-horizon regressor bundle loop (`t+30m`/`60m`/`120m`) | `src/models/cascade_and_gnn.py` | `train_conformal_cascade_models` |
| 125 | MLflow SQLite tracking URI configuration | `src/models/cascade_and_gnn.py` | `train_conformal_cascade_models` |
| 126 | MLflow experiment auto-creation (`railpulse-uk-cascade`) | `src/models/cascade_and_gnn.py` | `train_conformal_cascade_models` |
| 127 | MLflow parameter logging (`cv_strategy`, `confidence_level`) | `src/models/cascade_and_gnn.py` | `train_conformal_cascade_models` |
| 128 | MLflow metric logging (`mae`, `rmse`, `coverage_90`) | `src/models/cascade_and_gnn.py` | `train_conformal_cascade_models` |
| 129 | Serialised conformal model bundle persistence | `src/run_pipeline.py` | `run_end_to_end_pipeline` |
| 130 | Conformal model runtime inference loading | `src/app/main.py` | `_load_conformal_models` |
| 131 | Single-request 13-feature vector assembly | `src/app/main.py` | `predict_cascade` |
| 132 | Multi-horizon point + lower/upper API response | `src/app/main.py` | `predict_cascade` |
| 133 | Operational slice classifier (`Peak Commuter`, `Storm`, etc.) | `src/evaluation/error_analysis.py` | `_assign_operational_slice` |
| 134 | `Named Winter Storm` slice priority isolation | `src/evaluation/error_analysis.py` | `_assign_operational_slice` |
| 135 | `Signal/Infrastructure Failure` (`I1`/`I2`/`OT`) slicing | `src/evaluation/error_analysis.py` | `_assign_operational_slice` |
| 136 | `Peak Commuter` operational slice filtering | `src/evaluation/error_analysis.py` | `_assign_operational_slice` |
| 137 | `Normal` baseline operational slice filtering | `src/evaluation/error_analysis.py` | `_assign_operational_slice` |
| 138 | Empty-slice NaN-safe metric computation | `src/evaluation/error_analysis.py` | `_compute_slice_metrics` |
| 139 | Per-slice conformal 90% coverage audit | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 140 | Per-slice interval width sharpness comparison | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 141 | Multi-horizon conformal summary Markdown table | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 142 | Operational slice audit Markdown table | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 143 | Tail-delay threshold filtering (`>= 30.0 min`) | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 144 | Percentile fallback when extreme tail is sparse | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 145 | Station-hour alignment between LightGBM & GNN | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 146 | Model delta MAE computation (`gnn_mae - lgbm_mae`) | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 147 | Automated `docs/ERROR_ANALYSIS_REPORT.md` generation | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 148 | `ErrorAnalysisBundle` programmatic audit return | `src/evaluation/error_analysis.py` | `ErrorAnalysisBundle` |
| 149 | Reproducible random seeding across LightGBM & MAPIE | `src/models/cascade_and_gnn.py` | `train_conformal_cascade_models` |
| 150 | OpenMP thread-safety coordination on macOS arm64 | `src/__init__.py` | `os.environ.setdefault` |

---

## 4. PyTorch RailCascadeGAT Spatio-Temporal GNN (Skills 151–200)

| # | Capability | Implementation File | Function / Symbol |
|---:|---|---|---|
| 151 | Multi-head Graph Attention layer (`GraphAttentionLayer`) | `src/models/cascade_and_gnn.py` | `GraphAttentionLayer` |
| 152 | Learnable linear node feature projection (`self.W`) | `src/models/cascade_and_gnn.py` | `GraphAttentionLayer.__init__` |
| 153 | Learnable source attention parameter vector (`attn_src`) | `src/models/cascade_and_gnn.py` | `GraphAttentionLayer.__init__` |
| 154 | Learnable target attention parameter vector (`attn_dst`) | `src/models/cascade_and_gnn.py` | `GraphAttentionLayer.__init__` |
| 155 | Xavier uniform weight initialisation for GAT | `src/models/cascade_and_gnn.py` | `GraphAttentionLayer.__init__` |
| 156 | Pairwise additive attention score computation | `src/models/cascade_and_gnn.py` | `GraphAttentionLayer.forward` |
| 157 | `LeakyReLU(0.2)` attention non-linearity | `src/models/cascade_and_gnn.py` | `GraphAttentionLayer.forward` |
| 158 | Adjacency-masked attention (`-1e9` non-edge masking) | `src/models/cascade_and_gnn.py` | `GraphAttentionLayer.forward` |
| 159 | Support for 2D `(N, N)` and 3D `(B, N, N)` adjacency | `src/models/cascade_and_gnn.py` | `GraphAttentionLayer.forward` |
| 160 | Softmax neighbourhood attention normalisation | `src/models/cascade_and_gnn.py` | `GraphAttentionLayer.forward` |
| 161 | Attention weight dropout regularisation | `src/models/cascade_and_gnn.py` | `GraphAttentionLayer.forward` |
| 162 | Multi-head concatenation & output projection (`out_proj`) | `src/models/cascade_and_gnn.py` | `GraphAttentionLayer.forward` |
| 163 | Spatio-Temporal `RailCascadeGAT` network architecture | `src/models/cascade_and_gnn.py` | `RailCascadeGAT` |
| 164 | Input node feature `LayerNorm` stabilisation | `src/models/cascade_and_gnn.py` | `RailCascadeGAT.forward` |
| 165 | Two-layer message passing (`gat1`, `gat2`) | `src/models/cascade_and_gnn.py` | `RailCascadeGAT.forward` |
| 166 | Residual skip connection (`gat2(h1) + h1`) | `src/models/cascade_and_gnn.py` | `RailCascadeGAT.forward` |
| 167 | `GELU` hidden state activation | `src/models/cascade_and_gnn.py` | `RailCascadeGAT.forward` |
| 168 | Skip-concatenated node decoder (`cat([h2, x])`) | `src/models/cascade_and_gnn.py` | `RailCascadeGAT.forward` |
| 169 | `Softplus` non-negative multi-horizon delay output | `src/models/cascade_and_gnn.py` | `RailCascadeGAT.forward` |
| 170 | Simultaneous 3-horizon output (`t+30m`, `t+60m`, `t+120m`) | `src/models/cascade_and_gnn.py` | `RailCascadeGAT` |
| 171 | UK station adjacency matrix builder with self-loops | `src/models/cascade_and_gnn.py` | `build_station_adjacency` |
| 172 | Forward edge coupling + `0.5x` upstream feedback weights | `src/models/cascade_and_gnn.py` | `build_station_adjacency` |
| 173 | Deterministic alphabetical CRS index ordering | `src/models/cascade_and_gnn.py` | `build_station_adjacency` |
| 174 | Hourly spatio-temporal graph snapshot aggregation | `src/models/cascade_and_gnn.py` | `build_gnn_snapshot_tensors` |
| 175 | 8-channel station state tensor `(B, N, 8)` assembly | `src/models/cascade_and_gnn.py` | `build_gnn_snapshot_tensors` |
| 176 | Mean & peak station delay feature channels | `src/models/cascade_and_gnn.py` | `build_gnn_snapshot_tensors` |
| 177 | Mean turnaround slack station feature channel | `src/models/cascade_and_gnn.py` | `build_gnn_snapshot_tensors` |
| 178 | Scaled station PageRank centrality feature channel | `src/models/cascade_and_gnn.py` | `build_gnn_snapshot_tensors` |
| 179 | Max weather severity station feature channel | `src/models/cascade_and_gnn.py` | `build_gnn_snapshot_tensors` |
| 180 | Peak commuter ratio feature channel | `src/models/cascade_and_gnn.py` | `build_gnn_snapshot_tensors` |
| 181 | Infrastructure fault fraction (`I1`/`I2`/`OT`) channel | `src/models/cascade_and_gnn.py` | `build_gnn_snapshot_tensors` |
| 182 | Named Winter Storm indicator channel | `src/models/cascade_and_gnn.py` | `build_gnn_snapshot_tensors` |
| 183 | Multi-horizon snapshot target tensor `(B, N, 3)` | `src/models/cascade_and_gnn.py` | `build_gnn_snapshot_tensors` |
| 184 | Snapshot-to-telemetry alignment metadata DataFrame | `src/models/cascade_and_gnn.py` | `build_gnn_snapshot_tensors` |
| 185 | End-to-end PyTorch GAT training loop | `src/models/cascade_and_gnn.py` | `train_rail_cascade_gat` |
| 186 | `AdamW` optimiser with `1e-4` weight decay | `src/models/cascade_and_gnn.py` | `train_rail_cascade_gat` |
| 187 | `SmoothL1Loss` (`beta=2.0`) robust Huber regression | `src/models/cascade_and_gnn.py` | `train_rail_cascade_gat` |
| 188 | Tail-delay loss upweighting (`1.75x` for `>= 30 min`) | `src/models/cascade_and_gnn.py` | `train_rail_cascade_gat` |
| 189 | Active station observation masking in loss | `src/models/cascade_and_gnn.py` | `train_rail_cascade_gat` |
| 190 | Evaluation-mode `torch.no_grad()` inference pass | `src/models/cascade_and_gnn.py` | `train_rail_cascade_gat` |
| 191 | Station-hour `gnn_pred_30m`/`60m`/`120m` extraction | `src/models/cascade_and_gnn.py` | `train_rail_cascade_gat` |
| 192 | Overall GNN `t+60m` MAE & RMSE calculation | `src/models/cascade_and_gnn.py` | `train_rail_cascade_gat` |
| 193 | Tail-regime (`>= 30m`) GNN MAE computation | `src/models/cascade_and_gnn.py` | `train_rail_cascade_gat` |
| 194 | MLflow tracking for PyTorch `RailCascadeGAT` | `src/models/cascade_and_gnn.py` | `train_rail_cascade_gat` |
| 195 | Deterministic PyTorch & NumPy seed control | `src/models/cascade_and_gnn.py` | `train_rail_cascade_gat` |
| 196 | Single-threaded CPU PyTorch guard (`set_num_threads(1)`) | `src/models/cascade_and_gnn.py` | `torch.set_num_threads` |
| 197 | `GATTrainingResult` model & evaluation bundle | `src/models/cascade_and_gnn.py` | `GATTrainingResult` |
| 198 | Tail-delay GNN vs. LightGBM benchmark table | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 199 | Storm-regime GNN vs. LightGBM benchmark | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 200 | Infrastructure-failure GNN vs. LightGBM benchmark | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |

---

## 5. Guardrailed LangGraph NRCoT RAG & Delay Repay (Skills 201–250)

| # | Capability | Implementation File | Function / Symbol |
|---:|---|---|---|
| 201 | Disk-based NRCoT & TOC regulatory document loader | `src/agents/delay_repay_copilot.py` | `load_nrcot_regulatory_corpus` |
| 202 | PDF text extraction via `pypdf.PdfReader` | `src/agents/delay_repay_copilot.py` | `load_nrcot_regulatory_corpus` |
| 203 | Markdown (`.md`) & plain-text (`.txt`) regulatory parsing | `src/agents/delay_repay_copilot.py` | `load_nrcot_regulatory_corpus` |
| 204 | `[DOC-ID]` heading chunking & citation extraction | `src/agents/delay_repay_copilot.py` | `_parse_structured_sections` |
| 205 | Automatic regulatory topic classification | `src/agents/delay_repay_copilot.py` | `_infer_topic` |
| 206 | Graceful fallback to embedded statutory corpus | `src/agents/delay_repay_copilot.py` | `load_nrcot_regulatory_corpus` |
| 207 | Seeded NRCoT Conditions 14.2, 14.3, 28.1 & 32 doc | `data/regulatory_docs/nrcot_conditions_14_28_32.md` | `[NRCoT-14.2]` |
| 208 | Seeded Avanti West Coast 15-min charter doc | `data/regulatory_docs/avanti_west_coast_charter.md` | `[TOC-AVANTI-DR15]` |
| 209 | Seeded Northern Trains 15-min charter doc | `data/regulatory_docs/northern_trains_charter.md` | `[TOC-NORTHERN-DR15]` |
| 210 | Seeded LNER 30-min Passenger Charter doc | `data/regulatory_docs/lner_passenger_charter.md` | `[TOC-LNER-DR30]` |
| 211 | Seeded TransPennine Express 15-min charter doc | `data/regulatory_docs/transpennine_express_charter.md` | `[TOC-TPE-DR15]` |
| 212 | Seeded Network Rail Delay Attribution Guide doc | `data/regulatory_docs/network_rail_delay_attribution_guide.md` | `[NR-DAG-CODES]` |
| 213 | Hybrid BM25 + TF-IDF retriever class | `src/agents/delay_repay_copilot.py` | `HybridRegulationRetriever` |
| 214 | `BM25Okapi` lexical token scoring | `src/agents/delay_repay_copilot.py` | `HybridRegulationRetriever.search` |
| 215 | `TfidfVectorizer` unigram+bigram semantic matrix | `src/agents/delay_repay_copilot.py` | `HybridRegulationRetriever` |
| 216 | Cosine similarity ranking over TF-IDF vectors | `src/agents/delay_repay_copilot.py` | `HybridRegulationRetriever.search` |
| 217 | Min-max score normalisation & 50/50 hybrid fusion | `src/agents/delay_repay_copilot.py` | `HybridRegulationRetriever.search` |
| 218 | Alphanumeric regex tokenisation for clause codes | `src/agents/delay_repay_copilot.py` | `HybridRegulationRetriever._tokenize` |
| 219 | Canonical TOC alias normalisation (`WCML` -> `Avanti`) | `src/agents/delay_repay_copilot.py` | `normalise_toc_name` |
| 220 | Statutory TOC delay threshold lookup (`15` vs `30` min) | `src/agents/delay_repay_copilot.py` | `TOC_THRESHOLD_MIN` |
| 221 | TOC-to-charter citation mapping (`TOC_DOC_MAP`) | `src/agents/delay_repay_copilot.py` | `TOC_DOC_MAP` |
| 222 | Deterministic statutory Delay Repay calculator | `src/agents/delay_repay_copilot.py` | `calculate_statutory_delay_repay` |
| 223 | Avanti / Northern / TPE 15–29 min 25% single / 12.5% return | `src/agents/delay_repay_copilot.py` | `calculate_statutory_delay_repay` |
| 224 | LNER 30-min rule (0% under 30 minutes) | `src/agents/delay_repay_copilot.py` | `calculate_statutory_delay_repay` |
| 225 | 30–59 min 50% single / 25% return compensation tier | `src/agents/delay_repay_copilot.py` | `calculate_statutory_delay_repay` |
| 226 | 60–119 min 100% single / 50% return compensation tier | `src/agents/delay_repay_copilot.py` | `calculate_statutory_delay_repay` |
| 227 | 120+ min 100% full single or return compensation tier | `src/agents/delay_repay_copilot.py` | `calculate_statutory_delay_repay` |
| 228 | NRCoT Condition 14.2 split-ticket calling-pattern check | `src/agents/delay_repay_copilot.py` | `calculate_statutory_delay_repay` |
| 229 | NRCoT Condition 14.3 zonal/season ticket exception | `src/agents/delay_repay_copilot.py` | `calculate_statutory_delay_repay` |
| 230 | GBP estimated refund monetary calculation | `src/agents/delay_repay_copilot.py` | `calculate_statutory_delay_repay` |
| 231 | `CopilotState` TypedDict LangGraph state schema | `src/agents/delay_repay_copilot.py` | `CopilotState` |
| 232 | `StateGraph` cyclic/conditional LangGraph compilation | `src/agents/delay_repay_copilot.py` | `DelayRepayCopilot._build_graph` |
| 233 | Prompt-injection guardrail pattern detection | `src/agents/delay_repay_copilot.py` | `PROMPT_INJECTION_PATTERNS` |
| 234 | Guardrail & augmented hybrid retrieval node | `src/agents/delay_repay_copilot.py` | `DelayRepayCopilot._guardrail_and_retrieve_node` |
| 235 | Required statutory clause recall guarantee | `src/agents/delay_repay_copilot.py` | `DelayRepayCopilot._guardrail_and_retrieve_node` |
| 236 | Grounded decision drafting node with `[DOC-ID]` tags | `src/agents/delay_repay_copilot.py` | `DelayRepayCopilot._draft_decision_node` |
| 237 | Citation-verifier judge node | `src/agents/delay_repay_copilot.py` | `DelayRepayCopilot._citation_verifier_judge_node` |
| 238 | Hallucinated `[FAKE-ID]` citation detection | `src/agents/delay_repay_copilot.py` | `DelayRepayCopilot._citation_verifier_judge_node` |
| 239 | Missing-citation rejection gate | `src/agents/delay_repay_copilot.py` | `DelayRepayCopilot._citation_verifier_judge_node` |
| 240 | Statutory percentage consistency check | `src/agents/delay_repay_copilot.py` | `DelayRepayCopilot._citation_verifier_judge_node` |
| 241 | Conditional LangGraph routing (`verified` vs `fallback`) | `src/agents/delay_repay_copilot.py` | `DelayRepayCopilot._route_after_verification` |
| 242 | Deterministic `statutory_fallback` LangGraph node | `src/agents/delay_repay_copilot.py` | `DelayRepayCopilot._statutory_fallback_node` |
| 243 | External draft adversarial verification harness | `src/agents/delay_repay_copilot.py` | `DelayRepayCopilot.verify_external_draft` |
| 244 | End-to-end Delay Repay claim audit API | `src/agents/delay_repay_copilot.py` | `DelayRepayCopilot.audit_claim` |
| 245 | Extraordinary weather (`X8`) non-refusal enforcement | `data/regulatory_docs/nrcot_conditions_14_28_32.md` | `[ORR-DISRUPTION-32]` |
| 246 | 28-day statutory Delay Repay claim window citation | `data/regulatory_docs/nrcot_conditions_14_28_32.md` | `[NRCoT-28.1]` |
| 247 | Split-ticket through-journey compensation protection | `data/regulatory_docs/nrcot_conditions_14_28_32.md` | `[NRCoT-28.1]` |
| 248 | Sorted grounded citation list in response payload | `src/agents/delay_repay_copilot.py` | `DelayRepayCopilot._citation_verifier_judge_node` |
| 249 | Retrieved evidence provenance returned to caller | `src/agents/delay_repay_copilot.py` | `DelayRepayCopilot._citation_verifier_judge_node` |
| 250 | `CopilotSummary` audit summary dataclass | `src/agents/delay_repay_copilot.py` | `CopilotSummary` |

---

## 6. MLOps, Serving, UI, Testing & CI/CD (Skills 251–300)

| # | Capability | Implementation File | Function / Symbol |
|---:|---|---|---|
| 251 | FastAPI application & OpenAPI metadata definition | `src/app/main.py` | `app` |
| 252 | `GET /health` service, DuckDB & feed readiness probe | `src/app/main.py` | `health_check` |
| 253 | Timezone-aware UTC health timestamps (`datetime.now(UTC)`) | `src/app/main.py` | `health_check` |
| 254 | `CascadePredictionRequest` Pydantic v2 schema | `src/app/main.py` | `CascadePredictionRequest` |
| 255 | `POST /predict-cascade` multi-horizon inference endpoint | `src/app/main.py` | `predict_cascade` |
| 256 | Unknown UK CRS HTTP `400` validation guard | `src/app/main.py` | `predict_cascade` |
| 257 | `simulate_feed_outage` graceful degradation switch | `src/app/main.py` | `predict_cascade` |
| 258 | `RAILPULSE_LIVE_FEED_AVAILABLE` env outage toggle | `src/app/main.py` | `predict_cascade` |
| 259 | Deterministic outage fallback regressor | `src/app/main.py` | `_degraded_fallback_forecast` |
| 260 | Downstream station propagation embedded in API response | `src/app/main.py` | `predict_cascade` |
| 261 | `DelayRepayAuditRequest` Pydantic v2 schema | `src/app/main.py` | `DelayRepayAuditRequest` |
| 262 | `POST /audit-delay-repay` LangGraph copilot endpoint | `src/app/main.py` | `audit_delay_repay` |
| 263 | Streamlit multi-tab operational control UI | `src/app/dashboard.py` | `st.tabs` |
| 264 | UK station PageRank & delay heatmap view | `src/app/dashboard.py` | `station_metrics` |
| 265 | Station x weather regime pivot delay matrix | `src/app/dashboard.py` | `pivot_table` |
| 266 | Geospatial UK rail hub map (`st.map`) | `src/app/dashboard.py` | `st.map` |
| 267 | Interactive multi-hop knock-on cascade simulator UI | `src/app/dashboard.py` | `propagate_knock_on_delay` |
| 268 | Interactive NRCoT Delay Repay Copilot form UI | `src/app/dashboard.py` | `DelayRepayCopilot.audit_claim` |
| 269 | Embedded `ERROR_ANALYSIS_REPORT.md` markdown viewer | `src/app/dashboard.py` | `st.markdown` |
| 270 | CLI pipeline entrypoint (`python -m src.run_pipeline`) | `src/run_pipeline.py` | `main` |
| 271 | `--live / --no-live` `BooleanOptionalAction` CLI flag | `src/run_pipeline.py` | `main` |
| 272 | Configurable `--rows` corridor event volume flag | `src/run_pipeline.py` | `main` |
| 273 | End-to-end pipeline orchestration (`run_end_to_end_pipeline`) | `src/run_pipeline.py` | `run_end_to_end_pipeline` |
| 274 | Processed Parquet snapshots (`enriched_telemetry`, `station_metrics`) | `src/run_pipeline.py` | `run_end_to_end_pipeline` |
| 275 | PEP 621 project & dependency packaging | `pyproject.toml` | `[project]` |
| 276 | Core dependency pins (`httpx`, `tenacity`, `pypdf`, `mapie`) | `pyproject.toml` | `dependencies` |
| 277 | Ruff lint & Python 3.12 target rules | `pyproject.toml` | `[tool.ruff]` |
| 278 | Pytest discovery & `pythonpath` configuration | `pyproject.toml` | `[tool.pytest.ini_options]` |
| 279 | Multi-service containerised API `Dockerfile` (`libgomp1`) | `Dockerfile` | `CMD` |
| 280 | `pgvector/pgvector:pg16` vector DB Compose service | `docker-compose.yml` | `services.pgvector` |
| 281 | PostgreSQL `pg_isready` Compose healthcheck | `docker-compose.yml` | `services.pgvector.healthcheck` |
| 282 | MLflow SQLite tracking server Compose service | `docker-compose.yml` | `services.mlflow` |
| 283 | FastAPI container service with volume mounts | `docker-compose.yml` | `services.api` |
| 284 | GitHub Actions CI workflow on `push`/`pull_request` | `.github/workflows/ci.yml` | `lint-and-test` |
| 285 | CI `uv` Python 3.12 deterministic environment setup | `.github/workflows/ci.yml` | `Install dependencies` |
| 286 | CI Ruff static analysis enforcement | `.github/workflows/ci.yml` | `Run Ruff lint checks` |
| 287 | CI live end-to-end pipeline execution gate | `.github/workflows/ci.yml` | `Run end-to-end pipeline` |
| 288 | CI `pytest -v` regression suite gate | `.github/workflows/ci.yml` | `Run pytest suite` |
| 289 | Git ignore hygiene for `.venv`, `mlruns`, raw/processed data | `.gitignore` | `.gitignore` |
| 290 | Test: temporal-order quarantine & DuckDB PII exclusion | `tests/test_contracts_and_errors.py` | `test_temporal_order_quarantine_and_pnr_hashing` |
| 291 | Test: HMAC-SHA256 PNR determinism & key sensitivity | `tests/test_contracts_and_errors.py` | `test_temporal_order_quarantine_and_pnr_hashing` |
| 292 | Test: `NetworkX` PageRank sum & turnaround slack | `tests/test_contracts_and_errors.py` | `test_networkx_cascade_centrality_and_propagation` |
| 293 | Test: conformal 90% intervals & MLflow SQLite creation | `tests/test_contracts_and_errors.py` | `test_conformal_forecasting_gat_and_error_analysis` |
| 294 | Test: PyTorch `RailCascadeGAT` shape & non-negativity | `tests/test_contracts_and_errors.py` | `test_conformal_forecasting_gat_and_error_analysis` |
| 295 | Test: sliced Markdown error report generation | `tests/test_contracts_and_errors.py` | `test_conformal_forecasting_gat_and_error_analysis` |
| 296 | Test: Avanti 15m vs LNER 30m & split-ticket rules | `tests/test_contracts_and_errors.py` | `test_langgraph_nrcot_citation_verification_and_split_tickets` |
| 297 | Test: hallucinated citation & override fallback gate | `tests/test_contracts_and_errors.py` | `test_langgraph_nrcot_citation_verification_and_split_tickets` |
| 298 | Test: FastAPI `/health`, `/audit-delay-repay` & outage fallback | `tests/test_contracts_and_errors.py` | `test_fastapi_endpoints_and_degraded_feed_fallback` |
| 299 | Test: mocked Open-Meteo + TfL live ingestion & degraded fallback | `tests/test_contracts_and_errors.py` | `test_live_uk_rail_connectors_and_degraded_fallback` |
| 300 | Test: disk `.md`/`.txt`/`.pdf` NRCoT corpus parsing via `pypdf` | `tests/test_contracts_and_errors.py` | `test_load_nrcot_regulatory_corpus_pdf_md_txt` |
