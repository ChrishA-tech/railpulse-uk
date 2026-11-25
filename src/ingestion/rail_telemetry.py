"""UK Rail TRUST / Darwin telemetry ingestion, Pandera validation, GDPR PNR hashing, and quarantine."""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import duckdb
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

DEFAULT_PNR_HMAC_KEY: Final[str] = os.environ.get(
    "RAILPULSE_PNR_HMAC_KEY",
    "railpulse-uk-gdpr-dev-hmac-key-2026",
)

# Valid UK National Rail stations: CRS -> TIPLOC
UK_STATION_CODES: Final[dict[str, str]] = {
    "EUS": "EUSTON",
    "BHM": "BHAMNWS",
    "CRE": "CREWE",
    "MAN": "MNCRPIC",
    "LIV": "LVRPLSH",
    "PRE": "PRESTON",
    "GLC": "GLGC",
    "KGX": "KNGX",
    "PBO": "PBORO",
    "YRK": "YORK",
    "LDS": "LEEDS",
    "NCL": "NWCSTLE",
    "EDB": "EDINBUR",
    "SHF": "SHEFFLD",
}

UK_STATION_METADATA: Final[dict[str, dict[str, object]]] = {
    "EUS": {"name": "London Euston", "tiploc": "EUSTON", "lat": 51.5282, "lon": -0.1337, "corridor": "WCML"},
    "BHM": {"name": "Birmingham New Street", "tiploc": "BHAMNWS", "lat": 52.4778, "lon": -1.8989, "corridor": "WCML"},
    "CRE": {"name": "Crewe", "tiploc": "CREWE", "lat": 53.0890, "lon": -2.4337, "corridor": "WCML"},
    "MAN": {"name": "Manchester Piccadilly", "tiploc": "MNCRPIC", "lat": 53.4774, "lon": -2.2309, "corridor": "TPE"},
    "LIV": {"name": "Liverpool Lime Street", "tiploc": "LVRPLSH", "lat": 53.4076, "lon": -2.9776, "corridor": "TPE"},
    "PRE": {"name": "Preston", "tiploc": "PRESTON", "lat": 53.7556, "lon": -2.7073, "corridor": "WCML"},
    "GLC": {"name": "Glasgow Central", "tiploc": "GLGC", "lat": 55.8591, "lon": -4.2581, "corridor": "WCML"},
    "KGX": {"name": "London King's Cross", "tiploc": "KNGX", "lat": 51.5319, "lon": -0.1233, "corridor": "ECML"},
    "PBO": {"name": "Peterborough", "tiploc": "PBORO", "lat": 52.5748, "lon": -0.2504, "corridor": "ECML"},
    "YRK": {"name": "York", "tiploc": "YORK", "lat": 53.9580, "lon": -1.0932, "corridor": "ECML"},
    "LDS": {"name": "Leeds", "tiploc": "LEEDS", "lat": 53.7949, "lon": -1.5474, "corridor": "TPE"},
    "NCL": {"name": "Newcastle", "tiploc": "NWCSTLE", "lat": 54.9686, "lon": -1.6174, "corridor": "ECML"},
    "EDB": {"name": "Edinburgh Waverley", "tiploc": "EDINBUR", "lat": 55.9520, "lon": -3.1899, "corridor": "ECML"},
    "SHF": {"name": "Sheffield", "tiploc": "SHEFFLD", "lat": 53.3781, "lon": -1.4622, "corridor": "XC"},
}

# Network Rail Delay Attribution Guide codes + ON_TIME sentinel
VALID_DELAY_ATTRIBUTION_CODES: Final[set[str]] = {
    "ON_TIME",  # On-time / sub-threshold running
    "TG",       # Traincrew / driver diagramming or availability
    "X8",       # External severe weather / named storm / adhesion
    "I1",       # Infrastructure: track / points / rail defect
    "I2",       # Signalling / track circuit / axle counter failure
    "M2",       # Rolling stock mechanical / traction fault
    "YI",       # Reactionary / knock-on delay from prior service
    "OT",       # Overhead line (OLE) power disruption
    "XD",       # External emergency / trespass
}

VALID_CORRIDORS: Final[set[str]] = {"WCML", "ECML", "TPE", "XC"}
VALID_WEATHER_REGIMES: Final[set[str]] = {
    "Normal",
    "Named Winter Storm",
    "Heavy Rain",
    "High Wind",
    "Frost/Ice",
}


def pnr_hmac_pseudonymise(pnr: str, secret_key: str | None = None) -> str:
    """Pseudonymise a passenger booking PNR using keyed HMAC-SHA256 for UK GDPR compliance."""
    if not isinstance(pnr, str) or not pnr.strip():
        raise ValueError("PNR must be a non-empty string.")
    key = (secret_key or DEFAULT_PNR_HMAC_KEY).encode("utf-8")
    normalized_pnr = pnr.strip().upper().encode("utf-8")
    return hmac.new(key, normalized_pnr, hashlib.sha256).hexdigest()


def _check_temporal_monotonicity(df: pd.DataFrame) -> pd.Series:
    """Ensure actual_arrival >= actual_departure_prev for every movement record."""
    return df["actual_arrival"].notna() & df["actual_departure_prev"].notna() & (
        df["actual_arrival"] >= df["actual_departure_prev"]
    )


def _check_crs_tiploc_match(df: pd.DataFrame) -> pd.Series:
    """Ensure TIPLOC matches the expected TIPLOC for the given UK CRS code."""
    expected_tiploc = df["crs_code"].map(UK_STATION_CODES)
    return expected_tiploc.notna() & (df["tiploc"] == expected_tiploc)


DarwinTrustSchema = pa.DataFrameSchema(
    columns={
        "train_uid": pa.Column(
            str,
            checks=pa.Check.str_matches(r"^[A-Z0-9]{4,10}$"),
            nullable=False,
        ),
        "rid": pa.Column(str, checks=pa.Check.str_length(min_value=6), nullable=False),
        "corridor": pa.Column(str, checks=pa.Check.isin(VALID_CORRIDORS), nullable=False),
        "crs_code": pa.Column(
            str,
            checks=pa.Check.isin(set(UK_STATION_CODES.keys())),
            nullable=False,
        ),
        "tiploc": pa.Column(
            str,
            checks=pa.Check.isin(set(UK_STATION_CODES.values())),
            nullable=False,
        ),
        "scheduled_arrival": pa.Column("datetime64[ns]", nullable=False, coerce=True),
        "actual_departure_prev": pa.Column("datetime64[ns]", nullable=False, coerce=True),
        "actual_arrival": pa.Column("datetime64[ns]", nullable=False, coerce=True),
        "delay_minutes": pa.Column(
            float,
            checks=[pa.Check.ge(-15.0), pa.Check.le(600.0)],
            nullable=False,
            coerce=True,
        ),
        "delay_code": pa.Column(
            str,
            checks=pa.Check.isin(VALID_DELAY_ATTRIBUTION_CODES),
            nullable=False,
        ),
        "weather_regime": pa.Column(
            str,
            checks=pa.Check.isin(VALID_WEATHER_REGIMES),
            nullable=False,
        ),
        "is_peak_commuter": pa.Column(bool, nullable=False, coerce=True),
        "turnaround_slack_min": pa.Column(
            float,
            checks=[pa.Check.ge(0.0), pa.Check.le(180.0)],
            nullable=False,
            coerce=True,
        ),
        "pnr_hash": pa.Column(
            str,
            checks=pa.Check.str_matches(r"^[0-9a-f]{64}$"),
            nullable=False,
        ),
    },
    checks=[
        pa.Check(
            _check_temporal_monotonicity,
            error="Temporal monotonicity violated: actual_arrival < actual_departure_prev",
            element_wise=False,
        ),
        pa.Check(
            _check_crs_tiploc_match,
            error="CRS and TIPLOC mismatch for UK station",
            element_wise=False,
        ),
    ],
    strict=False,
    coerce=True,
)


@dataclass
class IngestionResult:
    """Summary of a Darwin/TRUST telemetry ingestion and quarantine run."""

    clean_df: pd.DataFrame
    quarantine_df: pd.DataFrame
    valid_count: int
    quarantine_count: int
    duckdb_path: Path
    quarantine_path: Path


def _prepare_telemetry_frame(raw_df: pd.DataFrame, secret_key: str | None = None) -> pd.DataFrame:
    """Pseudonymise raw booking PNRs and normalise timestamp columns before validation."""
    df = raw_df.copy()

    if "booking_pnr" in df.columns:
        def _safe_hash(val: object) -> str:
            if pd.isna(val) or not str(val).strip():
                return "INVALID_PNR"
            return pnr_hmac_pseudonymise(str(val), secret_key=secret_key)

        df["pnr_hash"] = df["booking_pnr"].map(_safe_hash)
        df = df.drop(columns=["booking_pnr"])
    elif "pnr_hash" not in df.columns:
        df["pnr_hash"] = "INVALID_PNR"

    for col in ("scheduled_arrival", "actual_departure_prev", "actual_arrival"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "delay_minutes" in df.columns:
        df["delay_minutes"] = pd.to_numeric(df["delay_minutes"], errors="coerce")
    if "turnaround_slack_min" in df.columns:
        df["turnaround_slack_min"] = pd.to_numeric(df["turnaround_slack_min"], errors="coerce")

    return df


def _extract_failure_reasons(err: SchemaErrors, df_index: pd.Index) -> dict[object, list[str]]:
    """Map failing DataFrame indices to human-readable contract violation reasons."""
    reasons: dict[object, list[str]] = {}
    fc = err.failure_cases
    if fc is None or fc.empty:
        return reasons

    for _, row in fc.iterrows():
        idx = row.get("index")
        if pd.isna(idx) or idx not in df_index:
            continue
        col = row.get("column") or "dataframe"
        check = row.get("check") or "schema_violation"
        reasons.setdefault(idx, []).append(f"{col}: {check}")
    return reasons


def ingest_darwin_trust_telemetry(
    raw_df: pd.DataFrame,
    duckdb_path: str | Path = "data/processed/railpulse.duckdb",
    quarantine_path: str | Path = "data/quarantine/darwin_quarantine.parquet",
    secret_key: str | None = None,
    table_name: str = "telemetry_events",
) -> IngestionResult:
    """Validate Darwin/TRUST telemetry, quarantine invalid records to Parquet, and persist clean rows to DuckDB."""
    duckdb_path = Path(duckdb_path)
    quarantine_path = Path(quarantine_path)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)

    prepared_df = _prepare_telemetry_frame(raw_df, secret_key=secret_key)

    # First catch rows with null/unparseable timestamps or numeric coercion failures
    bad_indices: set[object] = set()
    reasons_by_idx: dict[object, list[str]] = {}

    for col in ("scheduled_arrival", "actual_departure_prev", "actual_arrival"):
        if col in prepared_df.columns:
            null_mask = prepared_df[col].isna()
            for idx in prepared_df.index[null_mask]:
                bad_indices.add(idx)
                reasons_by_idx.setdefault(idx, []).append(f"{col}: null_or_invalid_timestamp")

    # Validate remaining rows with Pandera lazy validation
    candidate_df = prepared_df.drop(index=list(bad_indices)) if bad_indices else prepared_df

    if not candidate_df.empty:
        try:
            DarwinTrustSchema.validate(candidate_df, lazy=True)
        except SchemaErrors as err:
            extracted = _extract_failure_reasons(err, candidate_df.index)
            for idx, r_list in extracted.items():
                bad_indices.add(idx)
                reasons_by_idx.setdefault(idx, []).extend(r_list)
            # Safety net if any failure case had no index
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

    # Final validation pass on clean_df to guarantee 100% schema compliance
    if not clean_df.empty:
        clean_df = DarwinTrustSchema.validate(clean_df, lazy=False)

    # Persist quarantine Parquet (convert timestamps to ISO/ns cleanly)
    quarantine_out = quarantine_df.copy()
    for col in ("scheduled_arrival", "actual_departure_prev", "actual_arrival"):
        if col in quarantine_out.columns:
            quarantine_out[col] = quarantine_out[col].astype("string")
    quarantine_out.to_parquet(quarantine_path, index=False)

    # Persist clean validated telemetry into DuckDB
    with duckdb.connect(str(duckdb_path)) as con:
        con.register("clean_telemetry_view", clean_df)
        con.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM clean_telemetry_view"
        )

    return IngestionResult(
        clean_df=clean_df.reset_index(drop=True),
        quarantine_df=quarantine_df.reset_index(drop=True),
        valid_count=len(clean_df),
        quarantine_count=len(quarantine_df),
        duckdb_path=duckdb_path,
        quarantine_path=quarantine_path,
    )
