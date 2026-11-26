"""Compare two RailPulse rules using archived Darwin York–Edinburgh movements."""

import argparse
from collections import Counter
from datetime import datetime, time
import gzip
import json
from pathlib import Path
from statistics import mean
import xml.etree.ElementTree as ET

ORIGIN, DESTINATION = "YORK", "EDINBUR"


def minute(value):
    parsed = time.fromisoformat(value)
    return parsed.hour * 60 + parsed.minute + parsed.second / 60


def events(paths):
    """Read provider-timestamped messages for the two public calling points."""
    for path in sorted(paths):
        with gzip.open(path, "rb") as stream:
            for line in stream:
                if not (b'tpl="YORK"' in line or b'tpl="EDINBUR"' in line):
                    continue
                message = ET.fromstring(line)
                timestamp = datetime.fromisoformat(message.attrib["ts"])
                for update in message.findall(".//{*}TS"):
                    rid = update.get("rid")
                    if not rid:
                        continue
                    for location in update.findall("{*}Location"):
                        station = location.get("tpl")
                        if station in (ORIGIN, DESTINATION):
                            yield timestamp, rid, update.get("ssd"), station, location


def backtest(updates):
    seen = {}
    destination_services = set()
    forecasts = {}
    arrivals = {}
    exclusions = Counter()
    for timestamp, rid, service_date, station, location in sorted(updates, key=lambda event: event[0]):
        if not service_date or timestamp.date().isoformat() != service_date:
            continue
        if station == DESTINATION:
            destination_services.add(rid)
            if location.get("pta") and rid not in forecasts:
                seen[rid] = minute(location.get("pta"))
            actual = location.find("{*}arr")
            if actual is not None and actual.get("at"):
                arrivals[rid] = (timestamp, minute(actual.get("at")))
            continue
        if rid in forecasts:
            continue
        departure = location.find("{*}dep")
        schedule = location.get("ptd")
        if departure is None or not departure.get("at") or not schedule:
            continue
        planned = minute(schedule)
        if not (9 * 60 <= planned < 13 * 60):
            continue
        actual = minute(departure.get("at"))
        forecasts[rid] = (timestamp, seen.get(rid), planned, actual)

    results = []
    for rid, (timestamp, destination_plan, planned, actual) in forecasts.items():
        # A downstream forecast must have a published destination schedule when issued.
        if destination_plan is None:
            reason = "no_destination_schedule_at_york" if rid in destination_services else "no_edinburgh_update_in_archive"
            exclusions[reason] += 1
            continue
        if not 90 <= destination_plan - planned <= 180:
            exclusions["outside_corridor_journey_window"] += 1
            continue
        outcome = arrivals.get(rid)
        issued = timestamp.hour * 60 + timestamp.minute + timestamp.second / 60
        if outcome is None or outcome[0] <= timestamp or outcome[1] <= max(actual, issued):
            exclusions["no_later_recorded_arrival"] += 1
            continue
        origin_delay = actual - planned
        destination_delay = outcome[1] - destination_plan
        # Freeze the same two rules used in the Typical simulation before reading outcomes.
        baseline = origin_delay
        recovery = max(0, .72 * origin_delay)
        results.append((timestamp.date().isoformat(), abs(baseline - destination_delay),
                        abs(recovery - destination_delay)))
    if not results:
        raise ValueError("No complete York–Edinburgh services in the supplied archive")
    return results, dict(exclusions)


def summary(rows):
    return {
        "services": len(rows),
        "baseline_mae_minutes": round(mean(row[1] for row in rows), 3),
        "recovery_mae_minutes": round(mean(row[2] for row in rows), 3),
        "recovery_wins": sum(row[2] < row[1] for row in rows),
        "baseline_wins": sum(row[1] < row[2] for row in rows),
        "ties": sum(row[1] == row[2] for row in rows),
        "worst_recovery_gap_minutes": round(max(row[2] - row[1] for row in rows), 3),
        "worst_recovery_error_minutes": round(max(row[2] for row in rows), 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Directory containing only the intended hourly .gz archive files")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        paths = list(args.archive.glob("*.gz"))
        if not paths:
            raise ValueError("No compressed archive files found")
        rows, exclusions = backtest(events(paths))
        dates = sorted({row[0] for row in rows})
        result = {
            "source": "National Rail Darwin Push Port observations via the iLoveTrains archive",
            "route": "York to Edinburgh Waverley",
            "departure_window_gmt": "09:00–12:59 scheduled departure",
            "files": sorted(path.name for path in paths),
            "metrics": summary(rows),
            "daily": {day: summary([row for row in rows if row[0] == day]) for day in dates},
            "exclusions": exclusions,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, ET.ParseError) as error:
        parser.exit(2, f"Darwin backtest failed: {error}\n")


if __name__ == "__main__":
    main()
