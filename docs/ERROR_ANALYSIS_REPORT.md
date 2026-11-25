# RailPulse UK — Error Analysis & Conformal Audit Report

## 1. Multi-Horizon Conformal LightGBM Summary (`GroupKFold` by Rail Corridor)

| horizon   |   mae_min |   rmse_min |   coverage_90 |   mean_interval_width_min |
|:----------|----------:|-----------:|--------------:|--------------------------:|
| t+30m     |     1.578 |      3.285 |        0.9842 |                     9.402 |
| t+60m     |     1.96  |      3.502 |        0.9871 |                    11.704 |
| t+120m    |     1.953 |      3.189 |        0.9857 |                    10.883 |

## 2. Operational Slice Audit (`t+60m` Conformal Regressor, Target 90% Coverage)

| operational_slice             |   count |   mae |   rmse |   coverage_90 |   mean_interval_width |
|:------------------------------|--------:|------:|-------:|--------------:|----------------------:|
| Peak Commuter                 |     160 | 1.857 |  2.333 |        0.9938 |                12.846 |
| Named Winter Storm            |      88 | 3.629 |  7.726 |        0.9432 |                14.384 |
| Signal/Infrastructure Failure |     122 | 2.295 |  3.261 |        0.9836 |                13.983 |
| Normal                        |     328 | 1.438 |  1.863 |        0.997  |                 9.579 |

## 3. Spatio-Temporal `RailCascadeGAT` vs. LightGBM Tail-Delay Comparison (`t+60m`)

| regime                        |   snapshots |   lgbm_mae_min |   gnn_mae_min |   lgbm_rmse_min |   gnn_rmse_min |   delta_mae_gnn_minus_lgbm |
|:------------------------------|------------:|---------------:|--------------:|----------------:|---------------:|---------------------------:|
| All Station-Hours             |         698 |          1.96  |         2.453 |           3.502 |          3.187 |                      0.493 |
| Tail Delays (>= 30m)          |         112 |          3.87  |         2.877 |           7.395 |          3.599 |                     -0.993 |
| Named Winter Storm            |          88 |          3.629 |         2.865 |           7.726 |          3.453 |                     -0.764 |
| Signal/Infrastructure Failure |         136 |          2.371 |         2.903 |           3.433 |          3.651 |                      0.532 |

## 4. Key Operational Findings

- **Conformal Calibration**: `CrossConformalRegressor` (`GroupKFold` by rail corridor) maintains near-90% nominal coverage across normal and peak commuter traffic while widening intervals during `Named Winter Storm` and `Signal/Infrastructure Failure` disruptions.
- **Tail-Delay Dynamics**: During severe multi-station disruptions (`>= 30 min`), message-passing across adjacent UK hubs (`CRE`, `BHM`, `MAN`, `YRK`, `EUS`) captures knock-on cascade amplification across interconnected corridors.
- **Actionable Guardrail**: When predicted upper conformal bound exceeds TOC Delay Repay thresholds (`15 min` for Avanti/Northern/TPE; `30 min` for LNER), proactive passenger disruption alerts and crew diagramming slack buffers should be triggered.
