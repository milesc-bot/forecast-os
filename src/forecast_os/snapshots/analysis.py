"""Pure-pandas analytics over stacked snapshot history.

These helpers operate on the ``history`` frames produced by
:meth:`~forecast_os.snapshots.store.SnapshotStore.history` (every snapshot
stacked, carrying an ``as_of`` column) and on plain actual panels, so they
are testable without touching disk or pyarrow — the module imports neither.

Three questions they answer:

- :func:`snapshot_evolution` — how one fixed target period's number moved
  week over week (e.g. "how did our Q4 booking number evolve").
- :func:`forecast_vs_actual` — the committed-forecast-vs-actual audit trail,
  joining each as-of forecast to the eventual actual for the same period.
- :func:`accuracy_over_time` — per-``as_of`` accuracy of those committed
  forecasts (magnitude and signed bias).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..core.types import ID_COL, TARGET_COL, TIME_COL
from ..evaluation.metrics import bias, mae, pct_bias

__all__ = ["snapshot_evolution", "forecast_vs_actual", "accuracy_over_time"]

#: Below this magnitude an actual is treated as zero for percentage safety.
_EPS = 1e-12


def snapshot_evolution(
    history: pd.DataFrame,
    target_ds: Any,
    value_col: str = "y",
    series: str | list[str] | None = None,
) -> pd.DataFrame:
    """How the value recorded for a FIXED ``target_ds`` moved across snapshots.

    For each ``as_of`` snapshot in ``history``, pick the row for period
    ``target_ds`` and report its ``value_col`` — panel history (``value_col='y'``)
    or forecast history (``value_col='yhat'``). Returns
    ``[unique_id, as_of, value]`` sorted by ``(unique_id, as_of)``, optionally
    filtered to the ``unique_id`` value(s) in ``series``.
    """
    ds = history[TIME_COL]
    if pd.api.types.is_datetime64_any_dtype(ds):
        target = pd.Timestamp(target_ds)
    else:
        target = target_ds
    sub = history[history[TIME_COL] == target]
    if series is not None:
        if isinstance(series, str):
            series = [series]
        sub = sub[sub[ID_COL].isin(series)]
    out = sub[[ID_COL, "as_of", value_col]].rename(columns={value_col: "value"})
    return out.sort_values([ID_COL, "as_of"], kind="stable").reset_index(drop=True)


def forecast_vs_actual(
    forecast_history: pd.DataFrame,
    actual_panel: pd.DataFrame,
    value_col: str = "yhat",
) -> pd.DataFrame:
    """Join each as-of committed forecast to the eventual actual for its period.

    Every ``(unique_id, ds)`` forecast in ``forecast_history`` (carrying an
    ``as_of`` column) is inner-joined to the actual ``y`` in ``actual_panel``,
    so forecasts for periods with no known actual yet (future ``ds``) are
    dropped — the committed-forecast-vs-actual audit trail. Returns
    ``[unique_id, as_of, ds, forecast, actual, error, abs_error, pct_error]``
    sorted by ``(unique_id, as_of, ds)`` where ``error = forecast - actual``
    and ``pct_error`` is ``error / actual`` (nan on ~zero actuals).
    """
    fh = forecast_history[[ID_COL, "as_of", TIME_COL, value_col]].copy()
    actual = actual_panel[[ID_COL, TIME_COL, TARGET_COL]].drop_duplicates(
        [ID_COL, TIME_COL]
    )
    merged = fh.merge(actual, on=[ID_COL, TIME_COL], how="inner")
    merged = merged.rename(columns={value_col: "forecast", TARGET_COL: "actual"})
    merged["error"] = merged["forecast"] - merged["actual"]
    merged["abs_error"] = merged["error"].abs()
    # nan-safe: a ~zero actual becomes NaN in the denominator, so pct_error is
    # NaN there instead of raising or yielding +/-inf.
    safe = merged["actual"].where(merged["actual"].abs() > _EPS)
    merged["pct_error"] = merged["error"] / safe
    cols = [
        ID_COL,
        "as_of",
        TIME_COL,
        "forecast",
        "actual",
        "error",
        "abs_error",
        "pct_error",
    ]
    return merged[cols].sort_values(
        [ID_COL, "as_of", TIME_COL], kind="stable"
    ).reset_index(drop=True)


def accuracy_over_time(forecast_vs_actual_frame: pd.DataFrame) -> pd.DataFrame:
    """Per-``as_of`` accuracy of committed forecasts vs their actuals.

    Consumes the output of :func:`forecast_vs_actual` and aggregates each
    ``as_of`` cohort into ``[as_of, n, mae, bias, pct_bias]`` using the
    :mod:`forecast_os.evaluation.metrics` functions. ``bias`` and ``pct_bias``
    are signed (positive = the cohort's forecasts ran high), exposing
    systematic over/under-forecasting that ``mae`` alone hides.
    """
    rows = []
    for as_of, g in forecast_vs_actual_frame.groupby("as_of", sort=True):
        y = g["actual"].to_numpy(dtype=float)
        yhat = g["forecast"].to_numpy(dtype=float)
        rows.append(
            {
                "as_of": as_of,
                "n": int(len(g)),
                "mae": mae(y, yhat),
                "bias": bias(y, yhat),
                "pct_bias": pct_bias(y, yhat),
            }
        )
    return pd.DataFrame(rows, columns=["as_of", "n", "mae", "bias", "pct_bias"])
