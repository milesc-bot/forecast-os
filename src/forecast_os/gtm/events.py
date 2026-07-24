"""The bridge from CRM-export shape to the forecast-os panel contract.

CRM exports (Salesforce opportunity reports, HubSpot deal exports, product
event logs) arrive as one row per event: duplicate dates are legal, months
with no activity are simply absent, and the entity hierarchy lives in
columns like ``team`` / ``rep``. :func:`to_panel` turns that shape into the
``(unique_id, ds, y)`` contract the engine speaks: ids are joined into a
hierarchy-ready ``unique_id``, dates are bucketed to their containing
``freq`` period's label, values are aggregated per period, and interior
gaps are filled.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Tick

from ..core.exceptions import DataContractError
from ..core.types import ID_COL, TARGET_COL, TIME_COL, validate_panel

__all__ = ["to_panel"]

_AGGS = ("sum", "mean", "count")


def _bucket_to_period_label(dates: pd.Series, freq: str) -> pd.Series:
    """Map datetimes to the on-offset label of their CONTAINING ``freq`` period.

    Tick (sub-daily/daily) frequencies floor directly. Anchored offsets
    bucket by the containing calendar period — the same grouping as
    ``resample(freq)`` — labeled at the period start for start-anchored
    offsets (``MS``, ``QS``, ``YS``; identical to a plain rollback) and at
    the normalized period end for end-anchored offsets (``ME``, ``QE``,
    ``YE``, ``W``). Rolling back end-anchored offsets would split one
    calendar period across two buckets (e.g. ``ME`` maps 2026-03-15 to
    2026-02-28 but 2026-03-31 to itself); bucketing by the containing
    period keeps every March date in the single 2026-03-31 bucket.
    """
    offset = to_offset(freq)
    if isinstance(offset, Tick):  # sub-daily/daily fixed frequencies floor directly
        return dates.dt.floor(offset)
    normalized = dates.dt.normalize()
    try:
        periods = normalized.dt.to_period(offset)
    except (AttributeError, TypeError, ValueError):
        # No period equivalent (start-anchored offsets like MS/QS/YS):
        # rollback already lands on the containing period's start.
        return normalized.map({d: offset.rollback(d) for d in normalized.unique()})
    starts = periods.dt.start_time
    if offset.is_on_offset(starts.iloc[0]):  # anchoring is uniform per offset
        return starts
    ends = periods.dt.end_time.dt.normalize()
    if offset.is_on_offset(ends.iloc[0]):
        return ends
    # Exotic offsets where neither period edge is on-offset: legacy rollback.
    return normalized.map({d: offset.rollback(d) for d in normalized.unique()})


def to_panel(
    records: pd.DataFrame,
    id_cols: str | Sequence[str],
    date_col: str,
    value_col: str | None = None,
    freq: str = "MS",
    agg: str = "sum",
    fill_value: float = 0.0,
    sep: str = "/",
) -> pd.DataFrame:
    """Aggregate event-level records into a contract-clean ``(unique_id, ds, y)`` panel.

    ``unique_id`` is the ``id_cols`` values joined by ``sep`` (so multi-level
    ids like team/rep stay reconcilable by hierarchy tooling), ``ds`` is
    ``date_col`` bucketed to its containing ``freq`` period's label — the
    period start for start-anchored frequencies (``MS``, ``QS``, ...) and
    the period end for end-anchored ones (``ME``, ``QE``, ...), matching
    ``resample(freq)`` totals — and ``y`` is ``value_col`` aggregated per
    (id, period). Periods between each series' first and last observation
    with no records are filled with ``fill_value``.

    Parameters
    ----------
    records : one row per event/opportunity; duplicate dates are legal.
    id_cols : column name(s) identifying the series; joined by ``sep``.
    date_col : event date column; parsed with ``pd.to_datetime``.
    value_col : column to aggregate. When ``None`` the records are counted
        and ``agg`` is ignored.
    freq : pandas offset alias for the panel period (default monthly starts).
    agg : one of ``"sum"``, ``"mean"``, ``"count"``. All three share skipna
        semantics: with ``value_col`` given, ``"count"`` counts non-null
        values (NaN-valued rows are excluded, exactly as ``"sum"`` and
        ``"mean"`` skip them). Row counting regardless of value is
        ``value_col=None``.
    fill_value : value for interior periods with no records (default 0.0).
    sep : separator joining ``id_cols`` into ``unique_id``.

    Returns
    -------
    A DataFrame with columns ``(unique_id, ds, y)`` that passes
    :func:`~forecast_os.core.types.validate_panel`.
    """
    if not isinstance(records, pd.DataFrame):
        raise DataContractError(
            f"expected a pandas DataFrame of records, got {type(records).__name__}"
        )
    if agg not in _AGGS:
        raise ValueError(f"unknown agg {agg!r}; expected one of {_AGGS}")
    if isinstance(id_cols, str):
        id_cols = [id_cols]
    id_cols = list(id_cols)
    needed = [*id_cols, date_col] + ([value_col] if value_col is not None else [])
    missing = [c for c in needed if c not in records.columns]
    if missing:
        raise DataContractError(f"missing column(s) {missing} in records")
    if len(records) == 0:
        raise DataContractError("records frame is empty")

    try:
        dates = pd.to_datetime(records[date_col])
    except (ValueError, TypeError) as exc:
        raise DataContractError(
            f"column {date_col!r} contains unparseable dates: {exc}"
        ) from exc

    uid = records[id_cols[0]].astype(str)
    for col in id_cols[1:]:
        uid = uid + sep + records[col].astype(str)

    work = pd.DataFrame({ID_COL: uid.to_numpy(), TIME_COL: _bucket_to_period_label(dates, freq)})
    if value_col is not None:
        work[TARGET_COL] = records[value_col].to_numpy()
    grouped = work.groupby([ID_COL, TIME_COL], sort=True)
    if value_col is None:
        y = grouped.size()  # pure row count: no values to be null
    elif agg == "count":
        y = grouped[TARGET_COL].count()  # non-null count: skipna like sum/mean
    else:
        y = grouped[TARGET_COL].agg(agg)
    y = y.astype(float).rename(TARGET_COL)

    frames = []
    for series_id, g in y.groupby(level=0):
        counts = g.droplevel(0)
        full = pd.date_range(counts.index.min(), counts.index.max(), freq=freq)
        counts = counts.reindex(full, fill_value=fill_value)
        frames.append(
            pd.DataFrame({ID_COL: series_id, TIME_COL: counts.index, TARGET_COL: counts.values})
        )
    return validate_panel(pd.concat(frames, ignore_index=True))
