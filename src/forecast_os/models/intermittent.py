"""Intermittent-demand models: Croston's method and TSB.

Both models target lumpy/sparse demand series — counts or amounts that are
zero in most periods (spare parts, B2B orders, long-tail SKUs) — where
classical smoothers collapse toward zero or chase spikes. Demand must be
nonnegative; a negative value raises :class:`ForecastOSError`.

Conventions (fixed smoothing constants, no optimization):

- :class:`Croston` runs simple exponential smoothing separately on the
  nonzero demand *sizes* and on the *inter-demand intervals* (in periods),
  updating both only at demand periods. It is initialized from the first
  demand: size = first nonzero value, interval = number of periods up to and
  including it. The forecast is ``size / interval``, flat over the horizon.
- :class:`TSB` (Teunter-Syntetos-Babai) smooths the demand *probability*
  every period (constant ``beta``) and the demand *size* at demand periods
  (constant ``alpha``). Warm-up values are the in-sample demand frequency and
  the mean nonzero size (a standard, mildly non-causal initialization). The
  forecast is ``probability * size``, flat over the horizon. Unlike Croston,
  TSB decays toward zero during long demand-free stretches.

In-sample one-step fitted arrays (NaN during warm-up) feed the residual-based
Gaussian prediction intervals of :class:`PerSeriesForecaster`. An all-zero
series is guarded to a flat forecast of 0.
"""

from __future__ import annotations

import numpy as np

from ..core.base import PerSeriesForecaster
from ..core.exceptions import ForecastOSError
from ..core.registry import register

__all__ = ["Croston", "TSB"]


def _check_demand(y: np.ndarray, name: str) -> None:
    if (y < 0).any():
        raise ForecastOSError(
            f"{name} models nonnegative demand (counts/amounts >= 0); "
            f"got a negative value {float(y[y < 0][0])}"
        )


def _check_smoothing(value: float, name: str) -> float:
    if not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return float(value)


@register("croston", family="statistical")
class Croston(PerSeriesForecaster):
    """Croston's method for intermittent demand (SES on sizes and intervals).

    Simple exponential smoothing with a fixed ``alpha`` is applied to the
    nonzero demand sizes and to the inter-demand intervals; both update only
    at demand periods. The point forecast is the demand rate
    ``size / interval``, constant across the horizon.
    """

    min_train_size = 4

    def __init__(self, alpha: float = 0.1):
        self.alpha = _check_smoothing(alpha, "alpha")

    def _fit_series(self, y: np.ndarray) -> dict:
        _check_demand(y, self.name)
        n = len(y)
        fitted = np.full(n, np.nan)
        nonzero = np.flatnonzero(y > 0)
        if nonzero.size == 0:
            return {"fitted": fitted, "rate_": 0.0, "size_": 0.0, "interval_": np.nan}
        a = self.alpha
        first = int(nonzero[0])
        size = float(y[first])
        interval = float(first + 1)
        last_demand = first
        for t in range(first + 1, n):
            fitted[t] = size / interval
            if y[t] > 0:
                size = a * y[t] + (1 - a) * size
                interval = a * (t - last_demand) + (1 - a) * interval
                last_demand = t
        return {
            "fitted": fitted,
            "rate_": size / interval,
            "size_": size,
            "interval_": interval,
        }

    def _predict_series(self, state: dict, h: int) -> np.ndarray:
        return np.full(h, state["rate_"])


@register("tsb", family="statistical")
class TSB(PerSeriesForecaster):
    """Teunter-Syntetos-Babai method for intermittent demand.

    The demand probability is smoothed every period with ``beta`` toward the
    demand indicator, and the demand size is smoothed with ``alpha`` at
    demand periods only. The point forecast is ``probability * size``,
    constant across the horizon; obsolescence (a long run of zeros) decays
    the probability and thus the forecast, which Croston cannot do.
    """

    min_train_size = 4

    def __init__(self, alpha: float = 0.1, beta: float = 0.1):
        self.alpha = _check_smoothing(alpha, "alpha")
        self.beta = _check_smoothing(beta, "beta")

    def _fit_series(self, y: np.ndarray) -> dict:
        _check_demand(y, self.name)
        n = len(y)
        fitted = np.full(n, np.nan)
        nonzero = np.flatnonzero(y > 0)
        if nonzero.size == 0:
            return {"fitted": fitted, "rate_": 0.0, "prob_": 0.0, "size_": 0.0}
        a, b = self.alpha, self.beta
        prob = nonzero.size / n
        size = float(y[nonzero].mean())
        for t in range(n):
            fitted[t] = prob * size
            demand = y[t] > 0
            prob = b * float(demand) + (1 - b) * prob
            if demand:
                size = a * y[t] + (1 - a) * size
        return {"fitted": fitted, "rate_": prob * size, "prob_": prob, "size_": size}

    def _predict_series(self, state: dict, h: int) -> np.ndarray:
        return np.full(h, state["rate_"])
