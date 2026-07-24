"""Point-in-time snapshot history: the pipeline audit trail.

Freeze each week's panel (or committed forecast) with a :class:`SnapshotStore`
— an append-only directory of parquet files plus a JSON manifest — then ask
how a fixed target period moved week over week (:func:`snapshot_evolution`),
how committed forecasts fared against the eventual actuals
(:func:`forecast_vs_actual`), and how that accuracy trends per as-of cohort
(:func:`accuracy_over_time`).

The store needs the optional ``[snapshots]`` extra (pyarrow); the analysis
helpers are pure pandas. Importing this package never requires pyarrow — the
guard fires only when a :class:`SnapshotStore` is constructed.
"""

from .analysis import accuracy_over_time, forecast_vs_actual, snapshot_evolution
from .store import SnapshotStore
from .waterfall import pipeline_waterfall, waterfall_summary

__all__ = [
    "SnapshotStore",
    "snapshot_evolution",
    "forecast_vs_actual",
    "accuracy_over_time",
    "pipeline_waterfall",
    "waterfall_summary",
]
