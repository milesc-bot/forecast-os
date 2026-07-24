"""Quota attainment probability and pipeline coverage.

Turns a probabilistic forecast (a ``predict(h, level=[...])`` frame) into
the number a revenue leader actually asks for: the probability that the
horizon total meets quota.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..core.exceptions import ForecastOSError
from ..core.types import ID_COL

__all__ = ["attainment_probability", "pipeline_coverage"]


def _quota_for(uid: str, quota) -> float:
    if isinstance(quota, pd.Series):
        quota = quota.to_dict()
    if isinstance(quota, dict):
        if uid not in quota:
            raise ForecastOSError(f"no quota provided for series {uid!r}")
        return float(quota[uid])
    return float(quota)


def attainment_probability(
    forecast_df: pd.DataFrame,
    quota: float | dict[str, float] | pd.Series,
    level: int = 80,
) -> pd.DataFrame:
    """Probability that each series' horizon total meets its quota.

    Sums the per-step forecast means; recovers each step's forecast standard
    deviation from the ``level``% interval as ``(hi - lo) / (2 z_level)``;
    combines steps in quadrature (``total_sigma = sqrt(sum sigma_i^2)``,
    i.e. steps are treated as independent — a documented simplification that
    understates the variance of positively autocorrelated forecast errors);
    then evaluates ``P(Normal(sum_mean, total_sigma) >= quota)``.

    Parameters
    ----------
    forecast_df : output of ``predict(h, level=[level])`` with columns
        ``unique_id, yhat, lo-{level}, hi-{level}``.
    quota : scalar applied to every series, or a dict / Series keyed by
        ``unique_id``.
    level : the confidence level whose interval columns encode step sigma.

    Returns
    -------
    One row per series: ``(unique_id, expected, quota, p_attain)``. A series
    with zero forecast sigma degenerates to ``p_attain`` of 1.0 when
    ``expected >= quota`` and 0.0 otherwise. NaN anywhere in a series'
    ``yhat`` / ``lo`` / ``hi`` values raises :class:`ForecastOSError` naming
    the series — a hole in the forecast must never silently collapse into a
    confident 0.0/1.0 probability.
    """
    lo_col, hi_col = f"lo-{level}", f"hi-{level}"
    missing = [c for c in (ID_COL, "yhat", lo_col, hi_col) if c not in forecast_df.columns]
    if missing:
        raise ForecastOSError(
            f"forecast_df is missing column(s) {missing}; call "
            f"predict(h, level=[{level}]) to produce the {level}% interval columns"
        )
    z = float(stats.norm.ppf(0.5 + level / 200))

    rows = []
    for uid, g in forecast_df.groupby(ID_COL, sort=True):
        vals = g[["yhat", lo_col, hi_col]].to_numpy(dtype=float)
        if np.isnan(vals).any():
            raise ForecastOSError(
                f"forecast for series {uid!r} contains NaN in "
                f"yhat/{lo_col}/{hi_col}; attainment probability is undefined "
                f"for an incomplete forecast"
            )
        expected = float(g["yhat"].sum())
        sigma_steps = (g[hi_col].to_numpy(float) - g[lo_col].to_numpy(float)) / (2.0 * z)
        total_sigma = float(np.sqrt(np.sum(sigma_steps**2)))
        q = _quota_for(uid, quota)
        if total_sigma > 0:
            p = float(stats.norm.sf(q, loc=expected, scale=total_sigma))
        else:
            p = 1.0 if expected >= q else 0.0
        rows.append({ID_COL: uid, "expected": expected, "quota": q, "p_attain": p})
    return pd.DataFrame(rows, columns=[ID_COL, "expected", "quota", "p_attain"])


def pipeline_coverage(
    pipeline_value: float | pd.Series | dict[str, float],
    quota: float | pd.Series | dict[str, float],
) -> float | pd.Series:
    """Pipeline coverage ratio: open pipeline divided by quota.

    Scalar/Series aware: dicts are coerced to Series keyed by series id;
    if either input is a Series the result is a Series (aligned by key,
    scalars broadcast); two scalars return a float. A zero scalar quota
    raises; zero entries in a Series quota follow pandas division semantics
    (``inf``).
    """
    if isinstance(pipeline_value, dict):
        pipeline_value = pd.Series(pipeline_value, dtype=float)
    if isinstance(quota, dict):
        quota = pd.Series(quota, dtype=float)
    if not isinstance(pipeline_value, pd.Series) and not isinstance(quota, pd.Series):
        if float(quota) == 0.0:
            raise ValueError("quota must be nonzero")
        return float(pipeline_value) / float(quota)
    return pipeline_value / quota
