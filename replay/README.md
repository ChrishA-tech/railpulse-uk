# RailPulse UK decision replay

[Open the interactive portfolio dashboard](https://chrisha-tech.github.io/replay/?project=railpulse-uk).

The replay stress-tests two heuristics on simulated corridors. No historical train movements are included. GAT performance and conformal calibration require a dated train-movement archive.

## Reproduce

From the repository root, use Python 3.10+:

```sh
python replay/replay.py --output replay/example.json
python -m unittest discover -s replay -v
```

The committed HTML is a portable snapshot of the bundled report. Open `replay/index.html` directly. The dashboard supports group and scenario filters, case-level scores and CSV export.

## Error analysis on the bundled scenarios

Scores are in **Minutes absolute error**. Lower is better. Candidate regressions count cases where its score is higher.

| Scenario | Cases | Baseline mean | Candidate mean | Candidate regressions |
| --- | ---: | ---: | ---: | ---: |
| Typical | 30 | 3.767 | 2.066 | 10 |
| Disruption | 27 | 3.323 | 2.357 | 6 |
| Feed degraded | 27 | 4.560 | 4.560 | 0 |

The largest regression was 3.885 at case 29 (East Coast, Typical). The scenarios test software behaviour. They cannot establish real-world accuracy or savings.

## Replay an external archive

Supply a CSV with columns `period,group,scenario,baseline,candidate,actual`. Sort and align predictions and outcomes upstream. Each period/group pair must be unique. Supply genuinely out-of-sample predictions computed from information available at the decision time. The CLI validates input and scores cases. It does not train or backtest the source model for you.

For cost replay, put the realised policy costs in baseline and candidate, and a numeric placeholder in actual. For spill replay, use probabilities in [0,1] and binary outcomes. Other projects use predictions and observed numeric outcomes in the units above.

```sh
python replay/replay.py --input archive.csv --source "Archive name and holdout period" --output report.json
```

## Historical York–Edinburgh backtest

The [Darwin pilot](DARWIN_BACKTEST.md) compares the same two forecasting rules against recorded train arrivals. The interactive scenario replay uses simulated cases.
