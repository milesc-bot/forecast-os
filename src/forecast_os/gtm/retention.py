"""Cohort retention: panel building and the shifted-beta-geometric model.

:func:`cohort_panel` turns an already-aggregated cohort table (cohort id,
integer period age, retained count or rate) into the ``(unique_id, ds, y)``
contract with ``ds`` as integer age and ``y`` as the retention fraction.

:class:`ShiftedBetaGeometric` is the Fader-Hardie sBG model ("How to
project customer retention", 2007): each customer churns geometrically
with a probability drawn from a Beta(alpha, beta), giving the churn
recursion ``p_1 = alpha / (alpha + beta)``,
``p_t = (beta + t - 2) / (alpha + beta + t - 1) * p_{t-1}`` and survival
``S(t) = S(t-1) - p_t``. Parameters are estimated per cohort by maximum
likelihood on the observed retention curve, with an empirical-Bayes-lite
fallback: cohorts too short for a stable MLE borrow POOLED parameters
fitted on the position-wise mean curve of all cohorts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..core.base import PerSeriesForecaster
from ..core.exceptions import DataContractError, ForecastOSError
from ..core.registry import register
from ..core.types import ID_COL, TARGET_COL, TIME_COL, validate_panel

__all__ = ["cohort_panel", "ShiftedBetaGeometric"]

_EPS = 1e-9


def cohort_panel(
    records: pd.DataFrame,
    cohort_col: str,
    period_col: str,
    retained_col: str,
    n_customers_col: str | None = None,
) -> pd.DataFrame:
    """Build a retention panel from an aggregated cohort table.

    Parameters
    ----------
    records : one row per (cohort, period age) with the retained count or rate.
    cohort_col : cohort identifier column (becomes ``unique_id``).
    period_col : integer period age column (becomes ``ds``).
    retained_col : retained count (with ``n_customers_col``) or retention
        fraction (without).
    n_customers_col : cohort size column; when given, ``y`` is
        ``retained / n_customers``.

    Returns
    -------
    A ``(unique_id, ds, y)`` panel with integer ``ds`` and ``y`` in [0, 1].
    Monotone-nonincreasing curves are NOT required (measurement noise
    happens), but out-of-range fractions raise :class:`DataContractError`.
    """
    if not isinstance(records, pd.DataFrame):
        raise DataContractError(
            f"expected a pandas DataFrame of cohort records, got {type(records).__name__}"
        )
    needed = [cohort_col, period_col, retained_col] + (
        [n_customers_col] if n_customers_col is not None else []
    )
    missing = [c for c in needed if c not in records.columns]
    if missing:
        raise DataContractError(f"missing column(s) {missing} in cohort records")

    y = pd.to_numeric(records[retained_col]).astype(float)
    if n_customers_col is not None:
        y = y / pd.to_numeric(records[n_customers_col]).astype(float)
    if ((y < -_EPS) | (y > 1.0 + _EPS)).any():
        raise DataContractError(
            f"column {retained_col!r} yields retention fractions outside [0, 1]; "
            f"pass n_customers_col for raw counts"
        )
    panel = pd.DataFrame(
        {
            ID_COL: records[cohort_col].astype(str).to_numpy(),
            TIME_COL: pd.to_numeric(records[period_col]).astype(int).to_numpy(),
            TARGET_COL: y.to_numpy(),
        }
    )
    return validate_panel(panel)


def _sbg_survival(alpha: float, beta: float, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Churn probabilities ``p[1..T]`` and survival ``S[0..T]`` for the sBG model."""
    p = np.zeros(horizon + 1)
    s = np.ones(horizon + 1)
    if horizon >= 1:
        p[1] = alpha / (alpha + beta)
        s[1] = 1.0 - p[1]
        for t in range(2, horizon + 1):
            p[t] = (beta + t - 2.0) / (alpha + beta + t - 1.0) * p[t - 1]
            s[t] = s[t - 1] - p[t]
    return p, s


def _strip_leading_one(y: np.ndarray) -> tuple[np.ndarray, bool]:
    """Split off an age-0 observation of 1.0 so values map to survival S(1..T)."""
    y = np.asarray(y, dtype=float)
    if len(y) and y[0] >= 1.0 - _EPS:
        return y[1:], True
    return y, False


def _sbg_mle(s_obs: np.ndarray) -> tuple[float, float, bool]:
    """MLE of (alpha, beta) on one retention curve; third element is success.

    Maximizes the standard sBG likelihood with fractional weights: the churn
    fraction in period ``t`` weights ``ln p_t`` and the final survivors
    weight ``ln S(T)``. Noisy up-ticks (negative churn increments) are
    clipped to zero weight. Optimized over log-parameters (Nelder-Mead) so
    alpha, beta stay positive; runaway or non-finite solutions are reported
    as failures so callers can fall back to pooled parameters.
    """
    sobs_full = np.concatenate([[1.0], np.asarray(s_obs, dtype=float)])
    weights = np.clip(-np.diff(sobs_full), 0.0, None)
    horizon = len(s_obs)
    s_last = float(sobs_full[-1])

    def nll(log_params: np.ndarray) -> float:
        if np.any(np.abs(log_params) > 12.0):
            return 1e12
        a, b = np.exp(log_params)
        p, s = _sbg_survival(a, b, horizon)
        ll = float(np.sum(weights * np.log(np.maximum(p[1:], 1e-300))))
        ll += s_last * float(np.log(max(s[horizon], 1e-300)))
        return -ll

    try:
        res = minimize(
            nll,
            x0=np.zeros(2),
            method="Nelder-Mead",
            options={"xatol": 1e-7, "fatol": 1e-11, "maxiter": 2000},
        )
    except (ValueError, FloatingPointError):
        return 1.0, 1.0, False
    if (
        not np.all(np.isfinite(res.x))
        or not np.isfinite(res.fun)
        or np.any(np.abs(res.x) > 10.0)
    ):
        return 1.0, 1.0, False
    a, b = np.exp(res.x)
    return float(a), float(b), True


@register("retention_sbg", family="gtm")
class ShiftedBetaGeometric(PerSeriesForecaster):
    """Fader-Hardie shifted-beta-geometric cohort retention model.

    Fits (alpha, beta) per cohort by MLE on the observed retention curve and
    forecasts future retention by projecting the model survival curve
    forward from the last observed value:
    ``yhat_k = y_last * S(T + k) / S(T)`` — monotone nonincreasing and in
    [0, 1] by construction.

    Input convention: each series is a retention curve at consecutive
    integer ages; a first value of 1.0 is treated as the age-0 anchor and
    the remaining values as ``S(1..T)``. Values must be fractions in
    [0, 1] — anything else raises at ``fit`` (this model is retention-only
    by design and refuses generic panels).

    Empirical-Bayes-lite pooling: ``fit`` first estimates POOLED parameters
    on the position-wise mean curve across all cohorts, then fits each
    cohort; a cohort with fewer than ``pooled_threshold`` observations, or
    whose MLE fails, uses the pooled parameters instead (so even a 2-point
    cohort forecasts sensibly). At least one series must still have
    ``min_train_size`` observations for the pooled fit to mean anything.
    """

    min_train_size = 3

    def __init__(self, pooled_threshold: int = 5):
        if pooled_threshold < 2:
            raise ValueError(f"pooled_threshold must be >= 2, got {pooled_threshold}")
        self.pooled_threshold = pooled_threshold

    def fit(self, df: pd.DataFrame) -> ShiftedBetaGeometric:
        """Pooled pre-pass, then the standard per-series path."""
        df = validate_panel(df)
        y_all = df[TARGET_COL].to_numpy(dtype=float)
        if np.any((y_all < -_EPS) | (y_all > 1.0 + _EPS)):
            raise DataContractError(
                f"{self.name} models retention fractions in [0, 1]; got values in "
                f"[{y_all.min():.3g}, {y_all.max():.3g}]. Build the panel with "
                f"forecast_os.gtm.cohort_panel."
            )
        # The sBG recursion indexes survival by integer age, so each cohort's
        # ds must be consecutive integers: a gapped curve would silently map
        # observations to the wrong ages.
        for uid, g in df.groupby(ID_COL, sort=True):
            ages = g[TIME_COL].to_numpy()
            if not (np.issubdtype(ages.dtype, np.number) and np.all(np.diff(ages) == 1)):
                raise DataContractError(
                    f"{self.name} requires each cohort's 'ds' to be consecutive "
                    f"integer ages (0, 1, 2, ...); cohort {uid!r} violates this. "
                    f"Build the panel with forecast_os.gtm.cohort_panel and fill "
                    f"missing ages first."
                )
        min_needed = self.min_train_size
        lengths = df.groupby(ID_COL).size()
        if int(lengths.max()) < min_needed:
            raise ForecastOSError(
                f"{self.name} requires at least {min_needed} observations in at "
                f"least one series to fit pooled parameters; longest series has "
                f"{int(lengths.max())}"
            )
        self.pooled_params_ = self._fit_pooled(df)
        # Short cohorts are legal here because the pooled fallback covers them;
        # relax the per-series floor for the base-class loop, then restore it.
        self.min_train_size = 1
        try:
            super().fit(df)
        finally:
            self.min_train_size = min_needed
        return self

    def cohort_params(self) -> pd.DataFrame:
        """Fitted per-cohort parameters as (unique_id, alpha, beta, pooled)."""
        self._check_is_fitted()
        rows = [
            {
                ID_COL: uid,
                "alpha": state["alpha_"],
                "beta": state["beta_"],
                "pooled": state["pooled_"],
            }
            for uid, state in self._series_state.items()
        ]
        return pd.DataFrame(rows, columns=[ID_COL, "alpha", "beta", "pooled"])

    # -- internals -----------------------------------------------------------

    def _fit_pooled(self, df: pd.DataFrame) -> tuple[float, float]:
        curves = [
            _strip_leading_one(g[TARGET_COL].to_numpy(dtype=float))[0]
            for _, g in df.groupby(ID_COL, sort=True)
        ]
        max_len = max((len(c) for c in curves), default=0)
        if max_len < 1:
            return 1.0, 1.0
        mat = np.full((len(curves), max_len), np.nan)
        for i, c in enumerate(curves):
            mat[i, : len(c)] = c
        pooled_curve = np.nanmean(mat, axis=0)
        alpha, beta, ok = _sbg_mle(pooled_curve)
        return (alpha, beta) if ok else (1.0, 1.0)

    def _fit_series(self, y: np.ndarray) -> dict:
        s_obs, stripped = _strip_leading_one(y)
        horizon = len(s_obs)
        pooled = len(y) < self.pooled_threshold or horizon < 2
        if not pooled:
            alpha, beta, ok = _sbg_mle(s_obs)
            pooled = not ok
        if pooled:
            alpha, beta = self.pooled_params_
        _, s = _sbg_survival(alpha, beta, horizon)
        sobs_full = np.concatenate([[1.0], s_obs])
        # one-step-ahead conditional fits: carry last observed survival forward
        # by the model's period-over-period retention ratio
        fitted = sobs_full[:-1] * (s[1:] / np.maximum(s[:-1], 1e-12))
        if stripped:
            fitted = np.concatenate([[np.nan], fitted])
        return {
            "fitted": fitted,
            "alpha_": alpha,
            "beta_": beta,
            "pooled_": pooled,
            "T_": horizon,
            "s_last_": float(sobs_full[-1]),
        }

    def _predict_series(self, state: dict, h: int) -> np.ndarray:
        horizon = state["T_"]
        _, s = _sbg_survival(state["alpha_"], state["beta_"], horizon + h)
        base = max(s[horizon], 1e-12)
        return state["s_last_"] * s[horizon + 1 :] / base
