"""Trailing-window anomaly detection over a segmented (unique_id, ds, y) panel.

:func:`detect_anomalies` scans each series for points that deviate sharply from
their recent *trailing* history and returns only the flagged rows — the tool a
revenue leader reaches for to catch "EMEA sql->won conversion dropped" without
eyeballing every segment.

Two methods share one flag rule (``|score| >= threshold``):

``"zscore"``
    Classic rolling z-score. For each point the ``expected`` value is the mean
    of the ``window`` observations *strictly before* it and the deviation unit
    is that window's standard deviation (population, ``ddof=0``); the current
    point is excluded from its own baseline so a spike cannot mask itself.

``"iqr"``
    A robust variant: ``expected`` is the trailing rolling median and the
    deviation unit is the inter-quartile range scaled to a Gaussian sigma
    (``IQR / 1.349``). Less sensitive to one wild point contaminating the
    window than the mean/std z-score.

Zero-variance history is handled deliberately: when the deviation unit comes
out ``0`` a *material* change scores ``+/-inf`` (a level shift off a
zero-spread baseline is infinitely many sigmas — flagged, at the top severity
band), while no change scores ``0`` (not flagged). What "zero spread" means
differs by method, and ``"iqr"`` is the looser of the two: ``"zscore"`` needs a
*perfectly flat* window, whereas ``"iqr"`` only needs a flat interquartile
*core*, so a window like ``[10, 10, 10, 10, 10, 1000]`` has ``IQR == 0`` and any
subsequent move off 10 — however small — scores ``inf``. That is the robust
estimator behaving as specified (the wild point is deliberately ignored), but
it does mean ``"iqr"`` flags off majority-constant windows that ``"zscore"``
lets pass. The first ``window`` points of each series lack a full trailing
window and are never flagged.
"""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd

from ..core.exceptions import DataContractError
from ..core.types import ID_COL, TIME_COL

__all__ = ["detect_anomalies"]

#: Supported detection methods.
METHODS = ("zscore", "iqr")

# IQR of a Gaussian equals ~1.349 sigma; dividing the IQR by this recovers a
# robust standard-deviation estimate comparable to the zscore method's scale.
_IQR_TO_SIGMA = 1.349

_FLOAT_COLS = ("value", "expected", "score")


def _trailing_expected_and_scale(
    values: np.ndarray, window: int, method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return (expected, scale) arrays from the trailing window, excluding self.

    ``.shift(1)`` moves each rolling statistic forward one step so the point at
    index ``t`` is compared against the window ending at ``t - 1``. The first
    ``window`` entries are ``NaN`` (insufficient history).
    """
    series = pd.Series(values, dtype=float)
    roll = series.rolling(window, min_periods=window)
    if method == "zscore":
        expected = roll.mean().shift(1)
        scale = roll.std(ddof=0).shift(1)
    else:  # iqr
        expected = roll.median().shift(1)
        q1 = roll.quantile(0.25).shift(1)
        q3 = roll.quantile(0.75).shift(1)
        scale = (q3 - q1) / _IQR_TO_SIGMA
    return expected.to_numpy(dtype=float), scale.to_numpy(dtype=float)


def _scores(values: np.ndarray, expected: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Signed deviation scores, with zero-scale (flat history) handled.

    Where ``scale`` is positive the score is ``(value - expected) / scale``.
    Where ``scale`` is exactly 0 a value equal to ``expected`` scores 0 and any
    other value scores ``+/-inf`` (sign from the deviation). Warm-up rows keep a
    ``NaN`` score (from ``NaN`` expected/scale) so they are never flagged.
    """
    diff = values - expected
    with np.errstate(divide="ignore", invalid="ignore"):
        score = diff / scale
    flat = scale == 0.0
    if flat.any():
        near = np.isclose(values, expected, rtol=1e-9, atol=1e-12)
        score = np.where(flat & near, 0.0, score)
        score = np.where(flat & ~near, np.copysign(np.inf, diff), score)
    return score


def _severity(abs_score: np.ndarray, threshold: float) -> np.ndarray:
    """Categorical severity for flagged points, in bands of the threshold."""
    return np.select(
        [abs_score >= 2.0 * threshold, abs_score >= 1.5 * threshold],
        ["critical", "high"],
        default="moderate",
    )


def detect_anomalies(
    df: pd.DataFrame,
    value_col: str = "y",
    by: Hashable | None = ID_COL,
    method: str = "zscore",
    window: int = 6,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Flag trailing-window anomalies per series and return only the flagged rows.

    Parameters
    ----------
    df : DataFrame
        Long panel with a ``ds`` time column, the ``value_col`` value column,
        and (unless ``by`` is ``None``) the ``by`` segment column.
    value_col : str, default ``"y"``
        Column holding the numeric series value.
    by : hashable or None, default ``"unique_id"``
        Segment column to group by; each group is scanned independently. Rows
        whose segment label is null are scanned as their own (NaN-keyed)
        segment rather than dropped — an unassigned owner/team/region is an
        everyday CRM state, and silently leaving those rows unmonitored would
        return a clean-looking empty result for a segment nobody checked. Pass
        ``None`` to treat the whole frame as a single series (the output then
        omits the segment column).
    method : {"zscore", "iqr"}, default ``"zscore"``
        Detection method (see module docstring).
    window : int, default 6
        Trailing window length; must be ``>= 2``.
    threshold : float, default 3.0
        Flag a point when ``|score| >= threshold``. Larger is less sensitive.

    Returns
    -------
    DataFrame with columns ``[<by>,] ds, value, expected, score, severity`` —
    one row per flagged point, sorted by segment then ``ds``. ``score`` is the
    signed z-score (positive = spike, negative = drop); ``severity`` is one of
    ``"moderate"``, ``"high"``, ``"critical"``. Empty (with those columns) when
    nothing is flagged.

    Raises
    ------
    DataContractError
        ``df`` is not a DataFrame, a required column is missing, or
        ``value_col`` is non-numeric / contains ``NaN``.
    ValueError
        ``method`` is unknown, ``window < 2``, or ``threshold <= 0``.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataContractError(
            f"detect_anomalies expects a pandas DataFrame, got {type(df).__name__}"
        )
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; known methods: {list(METHODS)}")
    if not isinstance(window, (int, np.integer)) or window < 2:
        raise ValueError(f"window must be an integer >= 2, got {window!r}")
    if not threshold > 0:
        raise ValueError(f"threshold must be > 0, got {threshold!r}")

    required = [value_col, TIME_COL] + ([] if by is None else [by])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataContractError(
            f"detect_anomalies requires column(s) {missing}; frame has "
            f"{list(df.columns)}"
        )
    try:
        value_numeric = df[value_col].astype(float)
    except (TypeError, ValueError) as exc:
        raise DataContractError(
            f"value column {value_col!r} must be numeric: {exc}"
        ) from exc
    if value_numeric.isna().any():
        n = int(value_numeric.isna().sum())
        raise DataContractError(
            f"value column {value_col!r} contains {n} NaN value(s); impute or "
            f"drop them before anomaly detection"
        )

    out_cols = ([] if by is None else [by]) + [
        TIME_COL,
        "value",
        "expected",
        "score",
        "severity",
    ]

    if by is None:
        groups: list[tuple[object, pd.DataFrame]] = [(None, df)]
    else:
        # Null segment keys get their own segment: pandas' default dropna=True
        # would discard every row with a null key, leaving that whole segment
        # unscanned and reporting an ordinary-looking empty result — while the
        # same function hard-errors on a NaN in the VALUE column.
        #
        # This groups on factorized codes rather than passing dropna=False,
        # because pandas 2.x raises "Categorical categories cannot be null" for
        # groupby(dropna=False) as soon as the key column holds a null (with
        # either sort setting). Factorizing with use_na_sentinel=False gives
        # null its own code on every supported pandas; the original label is
        # restored below, and nulls sort last as groupby would place them.
        codes, uniques = pd.factorize(df[by], use_na_sentinel=False)
        null_codes = [i for i in range(len(uniques)) if pd.isna(uniques[i])]
        real_codes = sorted(
            (i for i in range(len(uniques)) if not pd.isna(uniques[i])),
            key=lambda i: uniques[i],
        )
        groups = [(uniques[i], df[codes == i]) for i in real_codes + null_codes]

    frames = []
    for key, group in groups:
        group = group.sort_values(TIME_COL, kind="stable")
        values = group[value_col].to_numpy(dtype=float)
        expected, scale = _trailing_expected_and_scale(values, window, method)
        score = _scores(values, expected, scale)
        abs_score = np.abs(score)
        flag = abs_score >= threshold  # NaN -> False, inf -> True
        if not flag.any():
            continue
        sub = pd.DataFrame(
            {
                TIME_COL: group[TIME_COL].to_numpy(),
                "value": values,
                "expected": expected,
                "score": score,
                "severity": _severity(abs_score, threshold),
            }
        )
        if by is not None:
            sub.insert(0, by, key)
        frames.append(sub.loc[flag].reset_index(drop=True))

    if frames:
        return pd.concat(frames, ignore_index=True)[out_cols]
    # An empty result has to carry the SAME 'ds' dtype a flagged one would, or
    # pd.concat-ing per-segment results downgrades the column to object and
    # costs the caller the .dt accessor. Taking it from the input also keeps an
    # integer-age panel (as retention uses) integer rather than forcing it to
    # datetime; the computed columns keep their float64.
    empty = {}
    for c in out_cols:
        if c in _FLOAT_COLS:
            dtype: object = "float64"
        elif c == TIME_COL:
            dtype = df[TIME_COL].dtype
        else:
            dtype = "object"
        empty[c] = pd.Series(dtype=dtype)
    return pd.DataFrame(empty)
