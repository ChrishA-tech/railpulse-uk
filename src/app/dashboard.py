"""Streamlit UI for RailPulse UK: Corridor Delay Heatmap, Cascade Simulator, and Delay Repay Copilot."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.agents.delay_repay_copilot import DelayRepayCopilot
from src.analytics.network_cascade import (
    build_uk_rail_graph,
    compute_station_cascade_metrics,
    propagate_knock_on_delay,
)
from src.ingestion.rail_telemetry import UK_STATION_CODES

st.set_page_config(page_title="RailPulse UK", page_icon="🚆", layout="wide")
st.title("RailPulse UK — Delay Cascade & NRCoT Delay Repay Copilot")

graph = build_uk_rail_graph()
metrics_path = Path("data/processed/station_metrics.parquet")
telemetry_path = Path("data/processed/enriched_telemetry.parquet")

if metrics_path.exists():
    station_metrics = pd.read_parquet(metrics_path)
else:
    station_metrics = compute_station_cascade_metrics(graph)

tab_heatmap, tab_cascade, tab_copilot, tab_report = st.tabs(
    [
        "UK Corridor Delay Heatmap",
        "Knock-On Cascade Simulator",
        "NRCoT Delay Repay Copilot",
        "Error Analysis Report",
    ]
)

with tab_heatmap:
    st.subheader("UK Station PageRank Centrality & Observed Delay Heatmap")
    st.dataframe(station_metrics, use_container_width=True)

    if telemetry_path.exists():
        tel_df = pd.read_parquet(telemetry_path)
        pivot = tel_df.pivot_table(
            index="crs_code",
            columns="weather_regime",
            values="delay_minutes",
            aggfunc="mean",
        ).round(2)
        st.subheader("Mean Arrival Delay (min) by UK Station & Weather Regime")
        st.dataframe(pivot, use_container_width=True)

    map_df = station_metrics[["crs_code", "station_name", "lat", "lon", "mean_delay_min"]].rename(
        columns={"lat": "latitude", "lon": "longitude"}
    )
    st.map(map_df, latitude="latitude", longitude="longitude")

with tab_cascade:
    st.subheader("Multi-Hop NetworkX Knock-On Delay Propagation")
    col1, col2, col3 = st.columns(3)
    with col1:
        origin_crs = st.selectbox("Origin Delay Station (CRS)", sorted(UK_STATION_CODES.keys()), index=1)
    with col2:
        initial_delay = st.slider("Initial Primary Delay (min)", min_value=5, max_value=150, value=35)
    with col3:
        slack_min = st.slider("Rolling-Stock Turnaround Slack (min)", min_value=0, max_value=30, value=6)

    prop_df = propagate_knock_on_delay(
        origin_crs=origin_crs,
        initial_delay_min=float(initial_delay),
        turnaround_slack_min=float(slack_min),
        graph=graph,
    )
    st.dataframe(prop_df, use_container_width=True)

with tab_copilot:
    st.subheader("Guardrailed LangGraph NRCoT & TOC Passenger Charter Copilot")
    copilot = DelayRepayCopilot()
    c1, c2 = st.columns(2)
    with c1:
        toc = st.selectbox(
            "Train Operating Company (TOC)",
            ["Avanti West Coast", "Northern", "LNER", "TransPennine Express"],
        )
        delay_min = st.number_input("Arrival Delay (minutes)", min_value=0.0, max_value=360.0, value=38.0)
        ticket_type = st.selectbox("Ticket Type", ["single", "return"])
    with c2:
        is_split = st.checkbox("Split Ticket Journey (NRCoT Condition 14.2)", value=True)
        calls_at_split = st.checkbox("Train Calls at Split Station", value=True)
        ticket_price = st.number_input("Ticket Fare (£ GBP)", min_value=0.0, value=84.0)

    user_query = st.text_area(
        "Journey / Claim Query",
        value="Avanti West Coast delay with split ticket at Crewe during winter storm disruption",
    )
    if st.button("Audit Delay Repay Claim"):
        result = copilot.audit_claim(
            query=user_query,
            toc=toc,
            delay_minutes=float(delay_min),
            ticket_type=ticket_type,
            is_split_ticket=is_split,
            train_calls_at_split_station=calls_at_split,
            ticket_price_gbp=float(ticket_price),
        )
        st.success(result["answer"])
        st.json(result)

with tab_report:
    report_file = Path("docs/ERROR_ANALYSIS_REPORT.md")
    if report_file.exists():
        st.markdown(report_file.read_text(encoding="utf-8"))
    else:
        st.info("Run `python -m src.run_pipeline` to generate `docs/ERROR_ANALYSIS_REPORT.md`.")
