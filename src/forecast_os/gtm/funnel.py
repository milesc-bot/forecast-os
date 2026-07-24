"""Funnel analytics: stage volumes, conversion-rate series, and propagation.

:func:`stage_panel` counts funnel events per stage over time,
:func:`conversion_rates` turns consecutive stage counts into forecastable
rate series, and :func:`propagate` chains a top-of-funnel forecast through
fixed conversion rates into implied downstream volumes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..core.exceptions import DataContractError, ForecastOSError
from ..core.types import ID_COL, TARGET_COL, TIME_COL
from .events import to_panel

__all__ = ["stage_panel", "conversion_rates", "propagate"]


def stage_panel(
    records: pd.DataFrame,
    stage_col: str,
    date_col: str,
    freq: str = "MS",
    id_cols: Sequence[str] | None = None,
    sep: str = "/",
) -> pd.DataFrame:
    """Count funnel events per stage over time as a ``(unique_id, ds, y)`` panel.

    ``unique_id`` is the stage name, prefixed by any ``id_cols`` values
    (e.g. ``"east/lead"``), so per-team funnels remain separate series.
    Interior periods with no events are filled with zero counts.
    """
    cols = [*(id_cols or []), stage_col]
    return to_panel(records, id_cols=cols, date_col=date_col, freq=freq, sep=sep)


def conversion_rates(
    stage_df: pd.DataFrame, stages: Sequence[str], sep: str = "/"
) -> pd.DataFrame:
    """Stage-to-next-stage conversion-rate series from a stage-count panel.

    For each consecutive pair in ``stages`` (an ordered top-to-bottom list),
    emits a series named ``"{a}->{b}"`` (prefixed like ``"east/a->b"`` when
    the stage ids carry a ``sep`` prefix) whose ``y`` is the same-period
    ratio ``count(b, t) / count(a, t)``.

    Conventions (documented, deliberate):

    - Same-period ratio: no conversion lag is modeled, so a month where
      downstream volume outpaces upstream volume can yield a rate above 1.
    - A zero denominator (including 0/0) yields ``NaN`` — the rate is
      undefined when nothing entered the upstream stage that period.
    - Periods missing from one stage's series but present in the other are
      treated as zero counts.

    The output has ``(unique_id, ds, y)`` columns; ``y`` may contain NaN, so
    pass ``allow_missing=True`` if you re-validate it as a panel.
    """
    stages = list(stages)
    if len(stages) < 2:
        raise ValueError(f"stages must list at least two stages, got {stages}")

    # split "prefix/stage" ids into (prefix, stage); plain ids have prefix ""
    work = stage_df[[ID_COL, TIME_COL, TARGET_COL]].reset_index(drop=True).copy()
    split = work[ID_COL].astype(str).str.rpartition(sep)
    work["_prefix"], work["_stage"] = split[0], split[2]

    found = set(work["_stage"])
    unknown = [s for s in stages if s not in found]
    if unknown:
        raise ForecastOSError(
            f"stage(s) {unknown} not present in stage_df; found stages {sorted(found)}"
        )

    frames = []
    for prefix, in_prefix in work.groupby("_prefix", sort=True):
        by_stage = {
            stage: g.set_index(TIME_COL)[TARGET_COL]
            for stage, g in in_prefix.groupby("_stage")
        }
        for a, b in zip(stages[:-1], stages[1:]):
            if a not in by_stage or b not in by_stage:
                continue
            top, bottom = by_stage[a], by_stage[b]
            index = top.index.union(bottom.index)
            top = top.reindex(index, fill_value=0.0)
            bottom = bottom.reindex(index, fill_value=0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                rate = np.where(top.to_numpy() == 0, np.nan, bottom.to_numpy())
                rate = rate / np.where(top.to_numpy() == 0, 1.0, top.to_numpy())
            name = f"{a}->{b}" if prefix == "" else f"{prefix}{sep}{a}->{b}"
            frames.append(pd.DataFrame({ID_COL: name, TIME_COL: index, TARGET_COL: rate}))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values([ID_COL, TIME_COL], kind="stable").reset_index(drop=True)


def propagate(top_forecast: pd.DataFrame, rates: dict[str, float]) -> pd.DataFrame:
    """Chain a top-of-funnel forecast through fixed conversion rates.

    ``rates`` maps each downstream stage to its conversion rate from the
    *previous* stage, in funnel order (dict order is preserved); each stage
    becomes a column of implied volume, e.g. ``{"mql": 0.5, "sql": 0.4}``
    yields ``mql = 0.5 * top`` and ``sql = 0.4 * mql``.

    Deliberately simple v1: rates are constant over the horizon and applied
    within-period. Lagged conversion (deals converting in later periods) is
    future work.

    The top-of-funnel volume is read from ``yhat`` (a forecast frame) or,
    failing that, ``y`` (a plain panel).
    """
    if not rates:
        raise ValueError("rates is empty; provide at least one downstream stage")
    for stage, rate in rates.items():
        if not (0.0 <= float(rate) <= 1.0):
            raise ValueError(f"conversion rate for {stage!r} must be in [0, 1], got {rate}")

    if "yhat" in top_forecast.columns:
        volume = top_forecast["yhat"].astype(float)
    elif TARGET_COL in top_forecast.columns:
        volume = top_forecast[TARGET_COL].astype(float)
    else:
        raise DataContractError(
            "top_forecast must have a 'yhat' (forecast) or 'y' (panel) column"
        )

    out = top_forecast.copy()
    for stage, rate in rates.items():
        if stage in out.columns:
            raise ValueError(f"stage {stage!r} collides with an existing column")
        volume = volume * float(rate)
        out[stage] = volume
    return out
