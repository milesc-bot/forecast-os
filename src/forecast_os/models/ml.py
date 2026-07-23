"""Machine-learning forecaster: closed-form ridge regression on lag features.

``RidgeLag`` builds a per-series design matrix of lagged targets, optionally
augmented with Fourier seasonality terms, standardizes features and target
(centering replaces an explicit intercept, so nothing is penalized unfairly),
and solves the ridge normal equations with ``np.linalg.solve``. Multi-step
forecasts are recursive: each prediction is appended to the lag window while
the Fourier clock keeps advancing.
"""

from __future__ import annotations

import numpy as np

from ..core.base import PerSeriesForecaster
from ..core.registry import register

__all__ = ["RidgeLag"]


def _fourier_terms(t: np.ndarray, season_length: int, k: int) -> np.ndarray:
    """Fourier features ``sin/cos(2*pi*i*t/m)`` for i=1..k, shape (len(t), 2k)."""
    cols = []
    for i in range(1, k + 1):
        arg = 2.0 * np.pi * i * t / season_length
        cols.append(np.sin(arg))
        cols.append(np.cos(arg))
    return np.column_stack(cols)


@register("ridge_lag", family="ml")
class RidgeLag(PerSeriesForecaster):
    """Ridge autoregression on lagged values plus optional Fourier seasonality."""

    def __init__(
        self,
        lags: int = 14,
        alpha: float = 1.0,
        season_length: int | None = None,
        fourier_k: int = 3,
    ):
        if lags < 1:
            raise ValueError(f"lags must be >= 1, got {lags}")
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        self.lags = lags
        self.alpha = alpha
        self.season_length = season_length
        self.fourier_k = fourier_k
        self.min_train_size = int(lags) + 10

    def _has_fourier(self) -> bool:
        return bool(self.season_length and self.season_length > 1 and self.fourier_k > 0)

    def _fit_series(self, y: np.ndarray) -> dict:
        n, m = len(y), self.lags
        # Row for target y[t] holds [y[t-1], ..., y[t-m]]; targets run t = m..n-1.
        x = np.column_stack([y[m - 1 - j : n - 1 - j] for j in range(m)])
        if self._has_fourier():
            t = np.arange(m, n, dtype=float)
            x = np.hstack([x, _fourier_terms(t, self.season_length, self.fourier_k)])
        target = y[m:]

        x_mean, x_std = x.mean(axis=0), x.std(axis=0)
        x_std = np.where(x_std < 1e-12, 1.0, x_std)
        y_mean = float(target.mean())
        y_std = float(target.std())
        y_std = y_std if y_std > 1e-12 else 1.0

        xs = (x - x_mean) / x_std
        ys = (target - y_mean) / y_std
        p = xs.shape[1]
        w = np.linalg.solve(xs.T @ xs + self.alpha * np.eye(p), xs.T @ ys)

        fitted = np.concatenate([np.full(m, np.nan), (xs @ w) * y_std + y_mean])
        return {
            "w": w,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "window": y[-m:].copy(),
            "n_obs": n,
            "fitted": fitted,
        }

    def _predict_series(self, state: dict, h: int) -> np.ndarray:
        m = self.lags
        hist = list(state["window"])
        out = np.empty(h)
        for k in range(h):
            feats = np.asarray(hist[-m:][::-1], dtype=float)  # [lag-1, ..., lag-m]
            if self._has_fourier():
                t = np.array([state["n_obs"] + k], dtype=float)
                feats = np.concatenate(
                    [feats, _fourier_terms(t, self.season_length, self.fourier_k)[0]]
                )
            xs = (feats - state["x_mean"]) / state["x_std"]
            out[k] = float(xs @ state["w"]) * state["y_std"] + state["y_mean"]
            hist.append(out[k])
        return out
