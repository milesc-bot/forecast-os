"""GARCH(1,1) volatility modeling.

``GARCH11`` is a standalone estimator for a single returns array: conditional
maximum likelihood of ``sigma2_t = omega + alpha*r_{t-1}^2 + beta*sigma2_{t-1}``
(returns demeaned and z-scored internally for a well-conditioned optimization,
parameters reported back on the original scale). ``GARCHVolatility`` wraps it
as a registered forecaster whose ``yhat`` is the predicted per-period
conditional volatility of each series.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..core.base import PerSeriesForecaster
from ..core.exceptions import NotFittedError
from ..core.registry import register

__all__ = ["GARCH11", "GARCHVolatility"]

_MIN_OBS = 10
_MAX_PERSISTENCE = 0.999


def _sigma2_path(r2: np.ndarray, omega: float, alpha: float, beta: float, var0: float):
    sigma2 = np.empty(r2.size)
    sigma2[0] = var0
    for t in range(1, r2.size):
        sigma2[t] = omega + alpha * r2[t - 1] + beta * sigma2[t - 1]
    return np.maximum(sigma2, 1e-12)


def _nll(params: np.ndarray, r2: np.ndarray, var0: float) -> float:
    omega, alpha, beta = params
    sigma2 = _sigma2_path(r2, omega, alpha, beta, var0)
    nll = 0.5 * float(np.sum(np.log(sigma2) + r2 / sigma2))
    persistence = alpha + beta
    if persistence > _MAX_PERSISTENCE:
        nll += 1e6 * (persistence - _MAX_PERSISTENCE)
    return nll


class GARCH11:
    """GARCH(1,1) conditional-volatility model estimated by MLE."""

    def fit(self, returns: np.ndarray) -> GARCH11:
        """Estimate (omega, alpha, beta) on ``returns`` (demeaned internally)."""
        r = np.asarray(returns, dtype=float).ravel()
        if r.size < _MIN_OBS:
            raise ValueError(f"GARCH11 requires at least {_MIN_OBS} observations, got {r.size}")
        if not np.isfinite(r).all():
            raise ValueError("returns contain NaN or infinite values")
        self.mu_ = float(np.mean(r))
        r = r - self.mu_
        scale = float(np.std(r))
        if scale < 1e-12:
            raise ValueError("returns are (near-)constant; GARCH volatility is undefined")

        z2 = (r / scale) ** 2
        var0 = float(np.mean(z2))
        bounds = [(1e-8, 10.0 * var0), (0.0, 0.998), (0.0, 0.998)]
        starts = ((0.05, 0.05, 0.90), (0.10, 0.10, 0.80), (0.30, 0.02, 0.95))
        best = None
        for x0 in starts:
            res = minimize(
                _nll, np.asarray(x0), args=(z2, var0), method="L-BFGS-B", bounds=bounds
            )
            if best is None or res.fun < best.fun:
                best = res
        omega_z, alpha, beta = best.x
        self.omega_ = float(omega_z * scale**2)
        self.alpha_ = float(alpha)
        self.beta_ = float(beta)

        r2 = r * r
        sigma2 = _sigma2_path(r2, self.omega_, self.alpha_, self.beta_, float(np.var(r)))
        self.cond_vol_ = np.sqrt(sigma2)
        self._last_r2 = float(r2[-1])
        self._last_var = float(sigma2[-1])
        self._is_fitted = True
        return self

    def _check_is_fitted(self) -> None:
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError("GARCH11 is not fitted; call fit() first")

    def forecast_variance(self, h: int) -> np.ndarray:
        """h-step variance forecast, geometric decay toward the long-run level."""
        self._check_is_fitted()
        if not isinstance(h, (int, np.integer)) or h < 1:
            raise ValueError(f"h must be a positive integer, got {h!r}")
        persistence = min(self.alpha_ + self.beta_, _MAX_PERSISTENCE)
        long_run = self.omega_ / (1.0 - persistence)
        var_next = self.omega_ + self.alpha_ * self._last_r2 + self.beta_ * self._last_var
        k = np.arange(h, dtype=float)  # exponent k-1 for steps 1..h
        return long_run + persistence**k * (var_next - long_run)

    def forecast_volatility(self, h: int) -> np.ndarray:
        """h-step volatility forecast (sqrt of :meth:`forecast_variance`)."""
        return np.sqrt(self.forecast_variance(h))


@register("garch", family="financial")
class GARCHVolatility(PerSeriesForecaster):
    """GARCH(1,1) volatility forecaster: yhat is predicted conditional volatility.

    Each series is z-score standardized before MLE and the volatility path is
    rescaled back, so the model is safe on arbitrary scales (levels or returns).
    """

    min_train_size = _MIN_OBS

    def _fit_series(self, y: np.ndarray) -> dict:
        sd = float(np.std(y))
        if sd < 1e-12:  # constant series: zero volatility
            fitted = np.zeros(y.size)
            fitted[0] = np.nan
            return {"garch": None, "sd": sd, "fitted": fitted}
        z = (y - np.mean(y)) / sd
        garch = GARCH11().fit(z)
        fitted = sd * garch.cond_vol_
        fitted[0] = np.nan  # sigma2_0 is an unconditional stub, not a fit
        return {"garch": garch, "sd": sd, "fitted": fitted}

    def _predict_series(self, state: dict, h: int) -> np.ndarray:
        if state["garch"] is None:
            return np.zeros(h)
        return state["sd"] * state["garch"].forecast_volatility(h)
