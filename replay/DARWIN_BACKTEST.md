# York to Edinburgh historical backtest

This pilot uses recorded Darwin Push Port updates from [the public archive](https://ilovetrains.co.uk/darwin-push-port), 17–19 November 2026. It covers services scheduled to depart York from 09:00 to 12:59 GMT and subsequently call at Edinburgh Waverley.

At the first recorded York departure update, the benchmark freezes the Edinburgh scheduled arrival available in the archive. Both rules use the same York departure delay. The last-delay baseline carries it forward. The recovery heuristic uses 72% of that delay with a floor of zero. The target is the recorded Edinburgh arrival minus the frozen scheduled arrival. The outcome must occur after the forecast was issued. No rules were fitted to the test sample.

| Forecast method | Mean absolute error | Cases with lower error |
| --- | ---: | ---: |
| Last observed delay | 7.850 min | 15 of 20 |
| Schedule recovery | 13.678 min | 5 of 20 |

The recovery rule performed better on 18 November (2.633 vs 3.333 min MAE), but worse on both other days. Its largest error disadvantage was 22.4 minutes. See [the full aggregate report](darwin-results.json) for daily results and exclusion counts.

This is a three-day pilot on one corridor and one departure window. The archive includes York departures that did not call at Edinburgh, and trains without usable schedule or arrival updates cannot be scored. Results describe the two stated rules. They do not establish accuracy for the RailPulse graph model. Raw Darwin messages are kept outside this repository.

## Reproduce

Download the 08:00–15:00 GMT hourly files for each date from the archive. Save them as `YYYY-MM-DD-HH.gz` in an otherwise empty folder. Use the [aggregate report](darwin-results.json) for the exact file list.

```sh
python replay/darwin_backtest.py /path/to/archive --output /tmp/darwin-results.json
diff -u replay/darwin-results.json /tmp/darwin-results.json
python -m unittest discover -s replay -v
```
