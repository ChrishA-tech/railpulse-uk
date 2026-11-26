from datetime import datetime
import gzip
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from darwin_backtest import backtest, events, summary


def update(clock, station, xml):
    return datetime.fromisoformat("2026-11-19T" + clock + "+00:00"), "train-1", "2026-11-19", station, ET.fromstring(xml)


class DarwinBacktestTest(unittest.TestCase):
    def test_prediction_uses_schedule_available_at_york(self):
        records = [
            update("09:50:00", "EDINBUR", '<Location pta="12:00"><arr et="12:10"/></Location>'),
            update("10:10:05", "YORK", '<Location ptd="10:00"><dep at="10:10"/></Location>'),
            update("12:15:10", "EDINBUR", '<Location pta="12:05"><arr at="12:15"/></Location>'),
        ]
        rows, exclusions = backtest(reversed(records))
        self.assertEqual(exclusions, {})
        result = summary(rows)
        self.assertEqual(result["baseline_mae_minutes"], 5)
        self.assertEqual(result["recovery_mae_minutes"], 7.8)

    def test_arrival_already_occurred_when_prediction_issued(self):
        records = [
            update("09:50:00", "EDINBUR", '<Location pta="12:00"/>'),
            update("12:12:00", "YORK", '<Location ptd="10:00"><dep at="10:12"/></Location>'),
            update("12:13:00", "EDINBUR", '<Location pta="12:00"><arr at="12:11"/></Location>'),
        ]
        with self.assertRaisesRegex(ValueError, "No complete"):
            backtest(records)

    def test_xml_archive_parses_timestamped_station_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "test.gz"
            with gzip.open(file, "wt") as stream:
                stream.write('<Pport ts="2026-11-19T10:10:05+00:00" xmlns="urn:p"><uR><TS rid="train-1" ssd="2026-11-19"><Location tpl="YORK" ptd="10:00"><dep at="10:10"/></Location></TS></uR></Pport>\n')
            parsed = list(events([file]))
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0][1:4], ("train-1", "2026-11-19", "YORK"))


if __name__ == "__main__":
    unittest.main()
