"""Live UK rail corridor weather (Open-Meteo) and rail status/Darwin (TfL/Huxley2) connectors."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Final

import httpx
import pandas as pd
from pandera.errors import SchemaErrors
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.ingestion.rail_telemetry import (
    UK_STATION_CODES,
    UK_STATION_METADATA,
    DarwinTrustSchema,
    _extract_failure_reasons,
    _prepare_telemetry_frame,
    generate_synthetic_uk_rail_telemetry,
    write_quarantine_parquet,
)

logger = logging.getLogger(__name__)

OPEN_METEO_URL: Final[str] = "https://api.open-meteo.com/v1/forecast"
TFL_STATUS_URL: Final[str] = "https://api.tfl.gov.uk/Line/Mode/elizabeth-line,overground/Status"
HUXLEY2_DEPARTURES_URL: Final[str] = "https://huxley2.azurewebsites.net/departures/{crs}"

CORE_LIVE_HUBS: Final[tuple[str, ...]] = ("LIV", "MAN", "CRE", "BHM", "EUS", "YRK", "EDB")


def classify_uk_weather_regime(
    temperature_c: float,
    precipitation_mm: float,
    wind_gusts_kmh: float,
) -> str:
    """Map live Open-Meteo observations to Network Rail operational weather regimes."""
    if wind_gusts_kmh >= 75.0:
        return "Named Winter Storm"
    if wind_gusts_kmh >= 50.0:
        return "High Wind"
    if precipitation_mm >= 3.0:
        return "Heavy Rain"
    if temperature_c <= 1.0:
        return "Frost/Ice"
    return "Normal"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.4, min=0.4, max=2.5),
    retry=retry_if_exception_type((httpx.HTTPError, ValueError)),
    reraise=True,
)
def _fetch_open_meteo_uk_hubs(
    client: httpx.Client,
    stations: list[str] | None = None,
) -> dict[str, dict[str, float | str]]:
    """Fetch live Open-Meteo weather observations across UK rail hub coordinates."""
    hubs = stations or sorted(UK_STATION_METADATA.keys())
    lats = ",".join(str(UK_STATION_METADATA[crs]["lat"]) for crs in hubs)
    lons = ",".join(str(UK_STATION_METADATA[crs]["lon"]) for crs in hubs)

    resp = client.get(
        OPEN_METEO_URL,
        params={
            "latitude": lats,
            "longitude": lons,
            "current": "temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m",
        },
    )
    resp.raise_for_status()
    payload = resp.json()
    entries = payload if isinstance(payload, list) else [payload]

    weather_by_crs: dict[str, dict[str, float | str]] = {}
    for crs, entry in zip(hubs, entries, strict=False):
        current = entry.get("current", {})
        temp = float(current.get("temperature_2m", 8.0))
        precip = float(current.get("precipitation", 0.0))
        wind_speed = float(current.get("wind_speed_10m", 10.0))
        wind_gusts = float(current.get("wind_gusts_10m", wind_speed * 1.4))
        weather_by_crs[crs] = {
            "temperature_c": temp,
            "precipitation_mm": precip,
            "wind_speed_kmh": wind_speed,
            "wind_gusts_kmh": wind_gusts,
            "weather_regime": classify_uk_weather_regime(temp, precip, wind_gusts),
        }
    return weather_by_crs


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.4, min=0.4, max=2.5),
    retry=retry_if_exception_type((httpx.HTTPError, ValueError)),
    reraise=True,
)
def _fetch_uk_rail_status(
    client: httpx.Client,
    darwin_token: str | None = None,
) -> dict[str, Any]:
    """Query National Rail Darwin/Huxley2 (when DARWIN_TOKEN is set) and TfL open rail status."""
    status_summary: dict[str, Any] = {
        "source": "tfl_open_status",
        "disrupted_lines": [],
        "delay_boost_min": 0.0,
        "darwin_services": [],
    }

    tfl_resp = client.get(TFL_STATUS_URL)
    tfl_resp.raise_for_status()
    tfl_data = tfl_resp.json()

    if isinstance(tfl_data, list):
        for line in tfl_data:
            line_name = str(line.get("name", line.get("id", "unknown")))
            statuses = line.get("lineStatuses", [])
            for st_item in statuses:
                sev = int(st_item.get("statusSeverity", 10))
                desc = str(st_item.get("statusSeverityDescription", "Good Service"))
                if sev < 10:
                    status_summary["disrupted_lines"].append(f"{line_name}: {desc}")
        if status_summary["disrupted_lines"]:
            status_summary["delay_boost_min"] = 6.5

    if darwin_token:
        hux_url = HUXLEY2_DEPARTURES_URL.format(crs="EUS")
        hux_resp = client.get(hux_url, params={"accessToken": darwin_token, "expand": "true"})
        hux_resp.raise_for_status()
        hux_json = hux_resp.json()
        services = hux_json.get("trainServices") or []
        status_summary["source"] = "darwin_huxley2+tfl"
        status_summary["darwin_services"] = [
            {
                "std": svc.get("std"),
                "etd": svc.get("etd"),
                "operator": svc.get("operator"),
                "destination": (svc.get("destination") or [{}])[0].get("crs"),
            }
            for svc in services[:10]
        ]

    return status_summary


def _validate_and_quarantine(
    raw_df: pd.DataFrame,
    quarantine_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate records with DarwinTrustSchema and persist failing rows via write_quarantine_parquet()."""
    prepared_df = _prepare_telemetry_frame(raw_df)

    bad_indices: set[object] = set()
    reasons_by_idx: dict[object, list[str]] = {}

    for col in ("scheduled_arrival", "actual_departure_prev", "actual_arrival"):
        if col in prepared_df.columns:
            for idx in prepared_df.index[prepared_df[col].isna()]:
                bad_indices.add(idx)
                reasons_by_idx.setdefault(idx, []).append(f"{col}: null_or_invalid_timestamp")

    candidate_df = prepared_df.drop(index=list(bad_indices)) if bad_indices else prepared_df
    if not candidate_df.empty:
        try:
            DarwinTrustSchema.validate(candidate_df, lazy=True)
        except SchemaErrors as err:
            extracted = _extract_failure_reasons(err, candidate_df.index)
            for idx, r_list in extracted.items():
                bad_indices.add(idx)
                reasons_by_idx.setdefault(idx, []).extend(r_list)
            if not extracted:
                for idx in candidate_df.index:
                    bad_indices.add(idx)
                    reasons_by_idx.setdefault(idx, []).append("schema_validation_failed")

    if bad_indices:
        ordered_bad = [idx for idx in prepared_df.index if idx in bad_indices]
        quarantine_df = prepared_df.loc[ordered_bad].copy()
        quarantine_df["quarantine_reason"] = [
            "; ".join(dict.fromkeys(reasons_by_idx.get(idx, ["contract_violation"])))
            for idx in ordered_bad
        ]
        clean_df = prepared_df.drop(index=ordered_bad).copy()
    else:
        quarantine_df = prepared_df.iloc[0:0].copy()
        quarantine_df["quarantine_reason"] = pd.Series(dtype="object")
        clean_df = prepared_df.copy()

    if not clean_df.empty:
        clean_df = DarwinTrustSchema.validate(clean_df, lazy=False)

    write_quarantine_parquet(quarantine_df, quarantine_path=quarantine_path)
    return clean_df.reset_index(drop=True), quarantine_df.reset_index(drop=True)


def fetch_live_uk_rail_corridor_telemetry(
    n_rows: int = 720,
    corrupt_fraction: float = 0.03,
    quarantine_path: str | Path = "data/quarantine/darwin_quarantine.parquet",
    timeout_seconds: float = 8.0,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Query live Open-Meteo + TfL/Darwin feeds, validate with DarwinTrustSchema, and quarantine bad rows.

    Falls back gracefully with ``LIVE_RAIL_FEED_DEGRADED`` to deterministic offline corridor data
    if live network endpoints are unreachable or return errors.
    """
    darwin_token = os.environ.get("DARWIN_TOKEN")
    feed_status = "LIVE"
    weather_obs: dict[str, dict[str, float | str]] = {}
    rail_status: dict[str, Any] = {
        "source": "offline_fallback",
        "disrupted_lines": [],
        "delay_boost_min": 0.0,
    }

    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout_seconds)
    try:
        hub_list = sorted(UK_STATION_CODES.keys())
        weather_obs = _fetch_open_meteo_uk_hubs(http_client, stations=hub_list)
        rail_status = _fetch_uk_rail_status(http_client, darwin_token=darwin_token)
    except Exception as exc:
        feed_status = "LIVE_RAIL_FEED_DEGRADED"
        logger.warning(
            "LIVE_RAIL_FEED_DEGRADED: falling back to deterministic offline corridor telemetry (%s)",
            exc,
        )
    finally:
        if owns_client:
            http_client.close()

    raw_df = generate_synthetic_uk_rail_telemetry(
        n_rows=n_rows,
        corrupt_fraction=corrupt_fraction,
        seed=42,
        weather_overrides=weather_obs or None,
        status_delay_boost_min=float(rail_status.get("delay_boost_min", 0.0)),
    )

    clean_df, quarantine_df = _validate_and_quarantine(raw_df, quarantine_path=quarantine_path)
    clean_df.attrs["feed_status"] = feed_status
    clean_df.attrs["weather_observations"] = weather_obs
    clean_df.attrs["rail_status"] = rail_status
    clean_df.attrs["raw_count"] = len(raw_df)
    clean_df.attrs["valid_count"] = len(clean_df)
    clean_df.attrs["quarantine_count"] = len(quarantine_df)
    clean_df.attrs["quarantine_path"] = str(quarantine_path)
    return clean_df
