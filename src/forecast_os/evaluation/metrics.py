"""Forecast accuracy metrics.

Point metrics operate on aligned 1-D arrays. Percentage metrics (``mape``,
``smape``) return *fractions* (0.05 == 5%); ``smape`` is the symmetric variant
bounded to [0, 2]. Governance metrics (``bias``, ``pct_bias``,
``tracking_signal``) are *signed*: they expose the direction of systematic
under/over-forecasting that absolute-error metrics hide. Scaled metrics
(``mase``, ``rmsse``) additionally need the training series and a seasonality
``m`` (the M4/M5 convention).

Interval metrics (:func:`coverage`, :func:`winkler_score`, pinball, CRPS)
score ``lo``/``hi`` prediction-interval columns; lower is better everywhere
except ``coverage``, which should sit close to its nominal ``level/100``.

:func:`evaluate` scores a :func:`~forecast_os.evaluation.backtest.cross_validation`
output frame per series and per model. Interval metrics are requested through
the same ``metrics`` list (``"coverage"``, ``"winkler"``, ``"pinball"``,
``"crps"``) and discover each model's confidence levels from its
``{model}-lo-{level}`` / ``{model}-hi-{level}`` sibling columns.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from ..core.types import ID_COL, TARGET_COL, TIME_COL

__all__ = [
    "mae",
    "rmse",
    "mape",
    "smape",
    "bias",
    "pct_bias",
    "tracking_signal",
    "mase",
    "rmsse",
    "pinball_loss",
    "coverage",
    "winkler_score",
    "evaluate",
]


def _align(y, yhat) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float).ravel()
    yhat = np.asarray(yhat, dtype=float).ravel()
    if y.shape != yhat.shape:
        raise ValueError(f"shape mismatch: y {y.shape} vs yhat {yhat.shape}")
    if y.size == 0:
        raise ValueError("empty input")
    return y, yhat


def mae(y, yhat) -> float:
    """Mean absolute error."""
    y, yhat = _align(y, yhat)
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat) -> float:
    """Root mean squared error."""
    y, yhat = _align(y, yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mape(y, yhat) -> float:
    """Mean absolute percentage error as a fraction; zero targets are excluded."""
    y, yhat = _align(y, yhat)
    mask = np.abs(y) > 1e-12
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])))


def smape(y, yhat) -> float:
    """Symmetric MAPE as a fraction in [0, 2]; 0/0 terms count as 0."""
    y, yhat = _align(y, yhat)
    denom = (np.abs(y) + np.abs(yhat)) / 2.0
    diff = np.abs(y - yhat)
    terms = np.where(denom > 1e-12, diff / np.where(denom > 1e-12, denom, 1.0), 0.0)
    return float(np.mean(terms))


def bias(y, yhat) -> float:
    """Mean signed error ``mean(yhat - y)``, in the units of ``y``.

    Negative values mean the forecast runs low (sandbagging), positive means
    it runs high — the direction that ``mae``/``mape`` hide. Offsetting
    errors cancel, so pair it with a magnitude metric.
    """
    y, yhat = _align(y, yhat)
    return float(np.mean(yhat - y))


def pct_bias(y, yhat) -> float:
    """Signed relative bias ``sum(yhat - y) / sum(|y|)`` as a fraction.

    ``-0.08`` reads as "the forecast ran 8% low over the window". Returns
    ``nan`` when the actuals are all ~0 (denominator below 1e-12).
    """
    y, yhat = _align(y, yhat)
    denom = float(np.sum(np.abs(y)))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum(yhat - y) / denom)


def tracking_signal(y, yhat) -> float:
    """Cumulative signed error over the mean absolute error (bounded by ±n).

    ``sum(yhat - y) / mean(|yhat - y|)``: values near ``+n`` / ``-n`` mean
    every error shares one sign — a systematically high/low forecast that
    needs review. Returns ``nan`` for a perfect forecast (MAD below 1e-12).
    """
    y, yhat = _align(y, yhat)
    err = yhat - y
    mad = float(np.mean(np.abs(err)))
    if mad < 1e-12:
        return float("nan")
    return float(np.sum(err) / mad)


def _naive_scale(y_train: np.ndarray, m: int, squared: bool) -> float:
    y_train = np.asarray(y_train, dtype=float).ravel()
    if len(y_train) <= m:
        raise ValueError(f"training series (len {len(y_train)}) must be longer than m={m}")
    d = y_train[m:] - y_train[:-m]
    return float(np.mean(d**2)) if squared else float(np.mean(np.abs(d)))


def mase(y, yhat, y_train, m: int = 1) -> float:
    """Mean absolute scaled error vs the seasonal-naive in-sample forecast."""
    scale = _naive_scale(y_train, m, squared=False)
    if scale < 1e-12:
        return float("nan")
    return mae(y, yhat) / scale


def rmsse(y, yhat, y_train, m: int = 1) -> float:
    """Root mean squared scaled error (the M5 competition metric)."""
    scale = _naive_scale(y_train, m, squared=True)
    if scale < 1e-12:
        return float("nan")
    y, yhat = _align(y, yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2) / scale))


def pinball_loss(y, q_pred, q: float) -> float:
    """Quantile (pinball) loss for quantile forecast ``q_pred`` at level ``q``."""
    if not 0 < q < 1:
        raise ValueError(f"q must be in (0, 1), got {q}")
    y, q_pred = _align(y, q_pred)
    diff = y - q_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def coverage(y, lo, hi) -> float:
    """Empirical coverage: fraction of actuals inside [lo, hi]."""
    y, lo = _align(y, lo)
    _, hi = _align(y, hi)
    return float(np.mean((y >= lo) & (y <= hi)))


def winkler_score(y, lo, hi, level: float) -> float:
    """Mean Winkler (interval) score at confidence ``level`` (e.g. 80).

    Each observation scores the interval width ``hi - lo`` plus, when the
    actual falls outside ``[lo, hi]``, a penalty of ``2/a`` times the distance
    to the violated bound, where ``a = 1 - level/100``. Lower is better;
    narrow intervals that still cover win.
    """
    if not 0 < level < 100:
        raise ValueError(f"level must be in (0, 100), got {level}")
    y, lo = _align(y, lo)
    _, hi = _align(y, hi)
    a = 1.0 - level / 100.0
    score = hi - lo
    score = score + np.where(y < lo, (2.0 / a) * (lo - y), 0.0)
    score = score + np.where(y > hi, (2.0 / a) * (y - hi), 0.0)
    return float(np.mean(score))


_SIMPLE_METRICS: dict[str, Callable] = {
    "mae": mae,
    "rmse": rmse,
    "mape": mape,
    "smape": smape,
    "bias": bias,
    "pct_bias": pct_bias,
    "tracking_signal": tracking_signal,
}
_SCALED_METRICS: dict[str, Callable] = {"mase": mase, "rmsse": rmsse}
_INTERVAL_METRICS = ("coverage", "winkler", "pinball", "crps")

_META_COLS = {ID_COL, TIME_COL, TARGET_COL, "cutoff"}


def _model_columns(cv_df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in cv_df.columns
        if c not in _META_COLS and "-lo-" not in c and "-hi-" not in c
    ]


def _interval_levels(columns: Iterable[str], model: str) -> list[int]:
    """Levels implied by ``{model}-lo-{l}`` / ``{model}-hi-{l}`` column pairs."""
    columns = set(columns)
    prefix = f"{model}-lo-"
    levels = set()
    for c in columns:
        if c.startswith(prefix):
            suffix = c[len(prefix) :]
            if suffix.isdigit() and f"{model}-hi-{suffix}" in columns:
                levels.add(int(suffix))
    return sorted(levels)


def _score_interval(g: pd.DataFrame, col: str, lvl: int, metric: str) -> float:
    """One interval-metric value for model ``col`` at level ``lvl`` on group ``g``."""
    lo_col, hi_col = f"{col}-lo-{lvl}", f"{col}-hi-{lvl}"
    valid = g[[TARGET_COL, col, lo_col, hi_col]].dropna()
    if len(valid) == 0:
        return float("nan")
    y, lo, hi = valid[TARGET_COL], valid[lo_col], valid[hi_col]
    if metric == "coverage":
        return coverage(y, lo, hi)
    if metric == "winkler":
        return winkler_score(y, lo, hi, lvl)
    # pinball: mean of the losses at the two implied tail quantiles
    q_lo, q_hi = 0.5 - lvl / 200.0, 0.5 + lvl / 200.0
    return 0.5 * (pinball_loss(y, lo, q_lo) + pinball_loss(y, hi, q_hi))


def _score_crps(g: pd.DataFrame, col: str, levels: list[int]) -> float:
    """Quantile-approximated CRPS: mean of 2x pinball over all implied quantiles."""
    interval_cols = [f"{col}-{side}-{lvl}" for lvl in levels for side in ("lo", "hi")]
    valid = g[[TARGET_COL, col, *interval_cols]].dropna()
    if len(valid) == 0:
        return float("nan")
    y = valid[TARGET_COL]
    terms = []
    for lvl in levels:
        terms.append(2.0 * pinball_loss(y, valid[f"{col}-lo-{lvl}"], 0.5 - lvl / 200.0))
        terms.append(2.0 * pinball_loss(y, valid[f"{col}-hi-{lvl}"], 0.5 + lvl / 200.0))
    return float(np.mean(terms))


def evaluate(
    cv_df: pd.DataFrame,
    metrics: Iterable[str] = ("mae", "rmse", "smape"),
    train_df: pd.DataFrame | None = None,
    seasonality: int = 1,
    by: str | None = None,
) -> pd.DataFrame:
    """Score a cross-validation frame per (series, metric, model).

    ``cv_df`` must have columns ``unique_id, ds, cutoff, y`` plus one column
    per model. Scaled metrics (mase/rmsse) require ``train_df`` (the panel the
    models were cross-validated on) for the naive scaling term.

    ``by=None`` (the default) pools all cutoffs into one score per series.
    ``by="cutoff"`` scores every ``(unique_id, cutoff)`` group separately —
    the output gains a ``cutoff`` column — which works for point and interval
    metrics alike and requires ``cv_df`` to carry a ``cutoff`` column. Any
    other ``by`` value raises :class:`ValueError`.

    Interval metrics — ``coverage``, ``winkler``, ``pinball``, ``crps`` — read
    each model's ``{model}-lo-{level}`` / ``{model}-hi-{level}`` columns (as
    produced by ``cross_validation(..., level=[...])``). The first three emit
    one row per discovered level, named ``coverage-{l}`` / ``winkler-{l}`` /
    ``pinball-{l}``; ``pinball-{l}`` is the mean pinball loss at the two
    implied tail quantiles ``0.5 -/+ l/200``. ``crps`` emits a single row with
    the quantile-approximated CRPS: the mean over all implied quantiles (both
    tails of every level) of twice the pinball loss.
    """
    if by not in (None, "cutoff"):
        raise ValueError(f"unknown by={by!r}; expected None or 'cutoff'")
    for col in (ID_COL, TARGET_COL):
        if col not in cv_df.columns:
            raise ValueError(f"cv_df is missing required column {col!r}")
    if by == "cutoff" and "cutoff" not in cv_df.columns:
        raise ValueError(
            "by='cutoff' requires a 'cutoff' column in cv_df "
            "(as produced by cross_validation), but this frame has none"
        )
    model_cols = _model_columns(cv_df)
    if not model_cols:
        raise ValueError("cv_df has no model forecast columns")
    metrics = list(metrics)
    for name in metrics:
        if (
            name not in _SIMPLE_METRICS
            and name not in _SCALED_METRICS
            and name not in _INTERVAL_METRICS
        ):
            known = sorted(_SIMPLE_METRICS) + sorted(_SCALED_METRICS) + sorted(_INTERVAL_METRICS)
            raise ValueError(f"unknown metric {name!r}; known: {known}")
        if name in _SCALED_METRICS and train_df is None:
            raise ValueError(f"metric {name!r} requires train_df")

    levels_by_model = {col: _interval_levels(cv_df.columns, col) for col in model_cols}
    interval_requested = [m for m in metrics if m in _INTERVAL_METRICS]
    if interval_requested:
        missing = [col for col in model_cols if not levels_by_model[col]]
        if missing:
            raise ValueError(
                f"interval metrics {interval_requested} need lo/hi columns, but "
                f"model(s) {missing} have none; pass level=[...] to "
                f"cross_validation (e.g. cross_validation(..., level=[80, 95]))"
            )
    all_levels = sorted({lvl for levels in levels_by_model.values() for lvl in levels})

    if by is None:
        groups = [({ID_COL: uid}, g) for uid, g in cv_df.groupby(ID_COL, sort=True)]
    else:
        groups = [
            ({ID_COL: uid, "cutoff": cutoff}, g)
            for (uid, cutoff), g in cv_df.groupby([ID_COL, "cutoff"], sort=True)
        ]

    rows = []
    for meta, g in groups:
        y_train = None
        if train_df is not None:
            y_train = train_df.loc[
                train_df[ID_COL] == meta[ID_COL], TARGET_COL
            ].to_numpy(float)
        for metric in metrics:
            if metric == "crps":
                row: dict = {**meta, "metric": "crps"}
                for col in model_cols:
                    row[col] = _score_crps(g, col, levels_by_model[col])
                rows.append(row)
            elif metric in _INTERVAL_METRICS:
                for lvl in all_levels:
                    row = {**meta, "metric": f"{metric}-{lvl}"}
                    for col in model_cols:
                        if lvl in levels_by_model[col]:
                            row[col] = _score_interval(g, col, lvl, metric)
                        else:
                            row[col] = float("nan")
                    rows.append(row)
            else:
                row = {**meta, "metric": metric}
                for col in model_cols:
                    valid = g[[TARGET_COL, col]].dropna()
                    if len(valid) == 0:
                        row[col] = float("nan")
                    elif metric in _SIMPLE_METRICS:
                        row[col] = _SIMPLE_METRICS[metric](valid[TARGET_COL], valid[col])
                    else:
                        row[col] = _SCALED_METRICS[metric](
                            valid[TARGET_COL], valid[col], y_train, m=seasonality
                        )
                rows.append(row)
    return pd.DataFrame(rows)
