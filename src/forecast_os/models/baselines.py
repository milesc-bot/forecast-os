"""Baseline forecasters: naive, seasonal naive, drift, and window average.

These are the reference points every richer model must beat. Each supplies
the analytic forecast-error growth of its implied stochastic process via
``_predict_sigma`` so prediction intervals widen correctly with horizon.
"""

from __future__ import annotations

import numpy as np

from ..core.base import PerSeriesForecaster
from ..core.registry import register

__all__ = ["Naive", "SeasonalNaive", "Drift", "WindowAverage"]


@register("naive", family="baseline")
class Naive(PerSeriesForecaster):
    """Repeat the last observed value (random-walk forecast)."""

    def _fit_series(self, y: np.ndarray) -> dict:
        return {"fitted": np.concatenate([[np.nan], y[:-1]]), "last_": float(y[-1])}

    def _predict_series(self, state: dict, h: int) -> np.ndarray:
        return np.full(h, state["last_"])

    def _predict_sigma(self, state: dict, h: int) -> np.ndarray:
        # random-walk forecast variance grows linearly with horizon
        return state["_sigma"] * np.sqrt(np.arange(1, h + 1))


@register("seasonal_naive", family="baseline")
class SeasonalNaive(PerSeriesForecaster):
    """Repeat the value observed one season ago."""

    def __init__(self, season_length: int = 7):
        if season_length < 1:
            raise ValueError(f"season_length must be >= 1, got {season_length}")
        self.season_length = season_length
        self.min_train_size = season_length

    def _fit_series(self, y: np.ndarray) -> dict:
        m = self.season_length
        fitted = np.full(len(y), np.nan)
        fitted[m:] = y[:-m] if m < len(y) else []
        return {"fitted": fitted, "season_": y[-m:].copy()}

    def _predict_series(self, state: dict, h: int) -> np.ndarray:
        season = state["season_"]
        return season[np.arange(h) % len(season)]

    def _predict_sigma(self, state: dict, h: int) -> np.ndarray:
        # variance grows once per completed seasonal cycle
        m = len(state["season_"])
        return state["_sigma"] * np.sqrt(np.arange(h) // m + 1)


@register("drift", family="baseline")
class Drift(PerSeriesForecaster):
    """Last value plus the average historical slope (naive with drift)."""

    min_train_size = 2

    def _fit_series(self, y: np.ndarray) -> dict:
        n = len(y)
        fitted = np.full(n, np.nan)
        if n > 2:
            t = np.arange(2, n)
            fitted[2:] = y[1:-1] + (y[1:-1] - y[0]) / (t - 1)
        return {
            "fitted": fitted,
            "last_": float(y[-1]),
            "slope_": float((y[-1] - y[0]) / (n - 1)),
            "n_": n,
        }

    def _predict_series(self, state: dict, h: int) -> np.ndarray:
        return state["last_"] + state["slope_"] * np.arange(1, h + 1)

    def _predict_sigma(self, state: dict, h: int) -> np.ndarray:
        k = np.arange(1, h + 1)
        return state["_sigma"] * np.sqrt(k * (1 + k / (state["n_"] - 1)))


@register("window_average", family="baseline")
class WindowAverage(PerSeriesForecaster):
    """Mean of the last ``window`` observations, repeated across the horizon."""

    def __init__(self, window: int = 7):
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = window

    def _fit_series(self, y: np.ndarray) -> dict:
        n = len(y)
        fitted = np.full(n, np.nan)
        if n > 1:
            csum = np.concatenate([[0.0], np.cumsum(y)])
            t = np.arange(1, n)
            lo = np.maximum(0, t - self.window)
            fitted[1:] = (csum[t] - csum[lo]) / (t - lo)  # expanding until t >= window
        return {"fitted": fitted, "mean_": float(np.mean(y[-self.window :]))}

    def _predict_series(self, state: dict, h: int) -> np.ndarray:
        return np.full(h, state["mean_"])
