"""Forecast accuracy metrics.

Point metrics operate on aligned 1-D arrays. Percentage metrics (``mape``,
``smape``) return *fractions* (0.05 == 5%); ``smape`` is the symmetric variant
bounded to [0, 2]. Scaled metrics (``mase``, ``rmsse``) additionally need the
training series and a seasonality ``m`` (the M4/M5 convention).

:func:`evaluate` scores a :func:`~forecast_os.evaluation.backtest.cross_validation`
output frame per series and per model.
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
    "mase",
    "rmsse",
    "pinball_loss",
    "coverage",
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


_SIMPLE_METRICS: dict[str, Callable] = {
    "mae": mae,
    "rmse": rmse,
    "mape": mape,
    "smape": smape,
}
_SCALED_METRICS: dict[str, Callable] = {"mase": mase, "rmsse": rmsse}

_META_COLS = {ID_COL, TIME_COL, TARGET_COL, "cutoff"}


def _model_columns(cv_df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in cv_df.columns
        if c not in _META_COLS and "-lo-" not in c and "-hi-" not in c
    ]


def evaluate(
    cv_df: pd.DataFrame,
    metrics: Iterable[str] = ("mae", "rmse", "smape"),
    train_df: pd.DataFrame | None = None,
    seasonality: int = 1,
) -> pd.DataFrame:
    """Score a cross-validation frame per (series, metric, model).

    ``cv_df`` must have columns ``unique_id, ds, cutoff, y`` plus one column
    per model. Scaled metrics (mase/rmsse) require ``train_df`` (the panel the
    models were cross-validated on) for the naive scaling term.
    """
    for col in (ID_COL, TARGET_COL):
        if col not in cv_df.columns:
            raise ValueError(f"cv_df is missing required column {col!r}")
    model_cols = _model_columns(cv_df)
    if not model_cols:
        raise ValueError("cv_df has no model forecast columns")
    metrics = list(metrics)
    for name in metrics:
        if name not in _SIMPLE_METRICS and name not in _SCALED_METRICS:
            known = sorted(_SIMPLE_METRICS) + sorted(_SCALED_METRICS)
            raise ValueError(f"unknown metric {name!r}; known: {known}")
        if name in _SCALED_METRICS and train_df is None:
            raise ValueError(f"metric {name!r} requires train_df")

    rows = []
    for uid, g in cv_df.groupby(ID_COL, sort=True):
        y_train = None
        if train_df is not None:
            y_train = train_df.loc[train_df[ID_COL] == uid, TARGET_COL].to_numpy(float)
        for metric in metrics:
            row: dict = {ID_COL: uid, "metric": metric}
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
