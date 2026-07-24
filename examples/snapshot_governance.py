"""Snapshot governance: track how the forecast moved, and how good it was.

Simulates four weekly snapshots of a bookings panel plus the committed
forecast at each, then answers the two questions a CRO actually asks:
how did our number move week over week, and were our commits calibrated?

Run:  python examples/snapshot_governance.py   (needs the [snapshots] extra)
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import forecast_os as fos
from forecast_os.snapshots import (
    SnapshotStore,
    accuracy_over_time,
    forecast_vs_actual,
    snapshot_evolution,
)

rng = np.random.default_rng(11)
store = SnapshotStore(Path(tempfile.mkdtemp()))

# The "true" bookings history that keeps growing each week we snapshot.
base = fos.generate_series(n_series=1, length=40, freq="W", level=500.0,
                           trend=3.0, seasonality=13, season_amp=40.0, seed=4)
base["unique_id"] = "total"

for week, cutoff in enumerate(range(28, 32)):
    as_of = base["ds"].iloc[cutoff - 1]
    known = base.iloc[:cutoff].copy()                 # what we knew that week
    store.snapshot(known, as_of=as_of, kind="panel")
    fc = fos.get_model("auto_ets", season_length=13).fit(known).predict(6, level=[80])
    store.snapshot(fc, as_of=as_of, kind="forecast", label="weekly commit")

print(f"captured {len(store.as_of_dates('panel'))} weekly snapshots\n")

# 1. Week over week: how did the forecast for one target week evolve?
target = base["ds"].iloc[33]
evo = snapshot_evolution(store.history(kind="forecast"), target_ds=target,
                         value_col="yhat", series="total")
print(f"forecast for {target.date()}, as it was committed each week:")
print(evo.assign(value=evo["value"].round(0)).to_string(index=False))

# 2. Governance: each committed forecast vs the actual that landed
audit = forecast_vs_actual(store.history(kind="forecast"), base)
print("\naccuracy of each weekly commit (once actuals arrived):")
acc = accuracy_over_time(audit)
print(acc.to_string(index=False, float_format="{:.2f}".format))
