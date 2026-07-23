"""The classical Theta method (Assimakopoulos & Nikolopoulos, 2000).

Decomposes the (optionally deseasonalized) series into two theta lines —
the OLS linear trend (theta=0) and a curvature-amplified line (theta=2 by
default) — SES-forecasts the theta line, and recombines:
``(1/theta) * SES + (1 - 1/theta) * trend extrapolation``.
"""

from __future__ import annotations

import numpy as np

from ..core.base import PerSeriesForecaster
from ..core.registry import register
from .ets import _fit_ses

__all__ = ["Theta"]


@register("theta", family="statistical")
class Theta(PerSeriesForecaster):
    """Classical two-theta-line method with optional seasonal adjustment."""

    min_train_size = 3

    def __init__(self, season_length: int | None = None, theta: float = 2.0):
        if theta < 1:
            raise ValueError(f"theta must be >= 1, got {theta}")
        self.season_length = season_length
        self.theta = theta

    def _seasonal_index(self, y: np.ndarray) -> tuple[str, np.ndarray] | tuple[None, None]:
        """Per-position seasonal indices; multiplicative when safe, else additive."""
        m = self.season_length
        if not m or m <= 1 or len(y) < 2 * m:
            return None, None
        pos = np.arange(len(y)) % m
        pos_means = np.array([float(np.mean(y[pos == i])) for i in range(m)])
        overall = float(np.mean(y))
        if np.all(y > 0) and overall > 1e-12:
            index = pos_means / overall
            return "mul", np.where(np.abs(index) < 1e-12, 1.0, index)
        return "add", pos_means - overall

    def _fit_series(self, y: np.ndarray) -> dict:
        n = len(y)
        t = np.arange(n, dtype=float)
        mode, index = self._seasonal_index(y)
        if mode == "mul":
            x = y / index[np.arange(n) % len(index)]
        elif mode == "add":
            x = y - index[np.arange(n) % len(index)]
        else:
            x = y
        slope, intercept = np.polyfit(t, x, 1)
        line0 = intercept + slope * t
        q = self.theta * x + (1 - self.theta) * line0
        alpha, level, ses_fitted = _fit_ses(q)
        w = 1.0 / self.theta
        fitted = w * ses_fitted + (1 - w) * line0
        if mode == "mul":
            fitted = fitted * index[np.arange(n) % len(index)]
        elif mode == "add":
            fitted = fitted + index[np.arange(n) % len(index)]
        return {
            "fitted": fitted,
            "alpha_": alpha,
            "level_": level,
            "slope_": float(slope),
            "intercept_": float(intercept),
            "mode_": mode,
            "index_": index,
            "n_": n,
        }

    def _predict_series(self, state: dict, h: int) -> np.ndarray:
        k = np.arange(1, h + 1)
        n = state["n_"]
        line0 = state["intercept_"] + state["slope_"] * (n - 1 + k)
        w = 1.0 / self.theta
        forecast = w * state["level_"] + (1 - w) * line0
        mode = state["mode_"]
        if mode:
            s = state["index_"][(n - 1 + k) % len(state["index_"])]
            forecast = forecast * s if mode == "mul" else forecast + s
        return forecast

    def _predict_sigma(self, state: dict, h: int) -> np.ndarray:
        # The theta=0 trend line is deterministic, so only the SES component
        # (combination weight w = 1/theta) accumulates forecast-error
        # variance. Approximate by scaling the SES horizon-growth term by w,
        # using the SES alpha fitted on the theta line.
        w = 1.0 / self.theta
        k = np.arange(1, h + 1)
        return state["_sigma"] * np.sqrt(1 + (k - 1) * (state["alpha_"] * w) ** 2)
