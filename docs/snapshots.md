# Snapshot store: point-in-time history

Forecasting is judged over time: *was last quarter's commit calibrated? how
did our Q4 number move week over week? which team's forecast should we
trust?* Answering those needs the one thing a live CRM query cannot give you
— **history of what you believed, when you believed it.** The snapshot store
is that persistence layer (extra: `pip install "forecast-os-gtm[snapshots]"`).

```python
from forecast_os.snapshots import SnapshotStore

store = SnapshotStore("~/pipeline-history")

# every week, capture the current panel and your committed forecast
store.snapshot(panel, as_of="2026-07-06", kind="panel")
store.snapshot(forecast, as_of="2026-07-06", kind="forecast", label="Q4 commit")
# ... next week ...
store.snapshot(panel, as_of="2026-07-13", kind="panel")
store.snapshot(forecast, as_of="2026-07-13", kind="forecast", label="Q4 commit")
```

Snapshots are append-only parquet files under the store directory, indexed by
a manifest. `store.load(as_of=...)` reads one back; `store.history()` stacks
them all with an `as_of` column for analysis.

## Week-over-week

How a *fixed* future period's number evolved as new information arrived:

```python
from forecast_os.snapshots import snapshot_evolution

hist = store.history(kind="forecast")
snapshot_evolution(hist, target_ds="2026-12-01", value_col="yhat", series="total")
#  unique_id      as_of      value
#      total 2026-07-06  3_100_000
#      total 2026-07-13  3_240_000   ← the Q4 commit moved up $140k in a week
```

## Forecast-accuracy governance over time

The audit trail: each committed forecast against the actual that eventually
landed — the persisted version of the governance metrics.

```python
from forecast_os.snapshots import forecast_vs_actual, accuracy_over_time

audit = forecast_vs_actual(store.history(kind="forecast"), actuals_panel)
accuracy_over_time(audit)
#      as_of   n     mae     bias  pct_bias
# 2026-07-06  6  41_200  -18_400    -0.061   ← this week's commit ran 6% low
```

The analysis functions are pure pandas over `history()` frames, so they work
on any stacked snapshot table (no store or pyarrow needed to analyze).
