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
_DATE_UNITS = ("s", "ms", "us")


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
    span: str = "series",
    sep: str = "/",
    date_unit: str | None = None,
    date_format: str | None = None,
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
    date_col : event date column; parsed with ``pd.to_datetime`` using
        ``date_unit`` / ``date_format`` when given. Integer/float date
        columns REQUIRE one of the two hints — bare ``pd.to_datetime``
        reads ints as nanosecond epoch offsets, silently collapsing e.g.
        Stripe epoch seconds or GA4 ``YYYYMMDD`` ints into 1970 — so
        numeric dates without a hint raise :class:`DataContractError`.
    value_col : column to aggregate. When ``None`` the records are counted
        and ``agg`` is ignored.
    freq : pandas offset alias for the panel period (default monthly starts).
    agg : one of ``"sum"``, ``"mean"``, ``"count"``. All three share skipna
        semantics: with ``value_col`` given, ``"count"`` counts non-null
        values (NaN-valued rows are excluded, exactly as ``"sum"`` and
        ``"mean"`` skip them). Row counting regardless of value is
        ``value_col=None``.
    fill_value : value for interior periods with no records (default 0.0).
        ``NaN`` marks the gaps as missing instead of inventing zeros; the
        panel then validates with ``allow_missing=True`` and downstream
        consumers must impute (e.g. ``preprocessing``) before models that
        reject missing targets.
    span : ``"series"`` (default) fills each series over its own first-to-last
        period; ``"panel"`` aligns every series to the global period range —
        required for hierarchies (``reconciled``) where children must share an
        aligned ``ds`` index, and correct for additive metrics (a rep who had
        not started yet booked 0.0).
    sep : separator joining ``id_cols`` into ``unique_id``.
    date_unit : epoch unit of numeric timestamps — ``"s"`` (Stripe
        ``created``, Mixpanel ``time``), ``"ms"``, or ``"us"`` (GA4
        ``event_timestamp``). Parsed via ``pd.to_datetime(..., unit=...)``;
        digit strings (e.g. a CSV read with ``dtype=str``) are coerced
        numeric first. Mutually exclusive with ``date_format``.
    date_format : strftime format for formatted dates, e.g. ``"%Y%m%d"``
        for GA4 ``event_date``. Numeric input is coerced int -> str before
        parsing so ``20260115`` and ``20260115.0`` both read as 2026-01-15.

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
    if span not in ("series", "panel"):
        raise ValueError(f"unknown span {span!r}; expected 'series' or 'panel'")
    if date_unit is not None and date_format is not None:
        raise ValueError("pass date_unit or date_format, not both")
    if date_unit is not None and date_unit not in _DATE_UNITS:
        raise ValueError(f"unknown date_unit {date_unit!r}; expected one of {_DATE_UNITS}")
    if isinstance(id_cols, str):
        id_cols = [id_cols]
    id_cols = list(id_cols)
    needed = [*id_cols, date_col] + ([value_col] if value_col is not None else [])
    missing = [c for c in needed if c not in records.columns]
    if missing:
        raise DataContractError(f"missing column(s) {missing} in records")
    if len(records) == 0:
        raise DataContractError("records frame is empty")

    raw = records[date_col]
    numeric_dates = pd.api.types.is_integer_dtype(raw) or pd.api.types.is_float_dtype(raw)
    if numeric_dates and date_unit is None and date_format is None:
        raise DataContractError(
            f"column {date_col!r} has numeric dtype {raw.dtype}: integer/float dates "
            "are ambiguous (epoch seconds? milliseconds? YYYYMMDD?) and would be read "
            "as nanosecond offsets from 1970. Pass date_unit='s'|'ms'|'us' for epoch "
            "timestamps or date_format (e.g. '%Y%m%d') for formatted numeric dates"
        )
    try:
        if date_unit is not None:
            # to_numeric first so digit strings (CSV dtype=str) also parse
            dates = pd.to_datetime(pd.to_numeric(raw), unit=date_unit)
        elif date_format is not None:
            if numeric_dates:  # 20260115 / 20260115.0 -> "20260115" (NaN-safe)
                raw = raw.astype("Int64").astype("string")
            dates = pd.to_datetime(raw, format=date_format)
        else:
            dates = pd.to_datetime(raw)
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
    panel_range = None
    if span == "panel":
        all_ds = y.index.get_level_values(TIME_COL)
        panel_range = pd.date_range(all_ds.min(), all_ds.max(), freq=freq)
    for series_id, g in y.groupby(level=0):
        counts = g.droplevel(0)
        if panel_range is not None:
            full = panel_range
        else:
            full = pd.date_range(counts.index.min(), counts.index.max(), freq=freq)
        counts = counts.reindex(full, fill_value=fill_value)
        frames.append(
            pd.DataFrame({ID_COL: series_id, TIME_COL: counts.index, TARGET_COL: counts.values})
        )
    # a NaN fill deliberately marks gap periods as missing-for-imputation
    return validate_panel(
        pd.concat(frames, ignore_index=True), allow_missing=bool(pd.isna(fill_value))
    )
