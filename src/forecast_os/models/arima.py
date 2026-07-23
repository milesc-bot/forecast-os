"""ARIMA models estimated by conditional sum of squares (CSS).

``ARIMA(p, d, q)`` differences the series ``d`` times, runs the innovation
recursion ``e_t = w_t - c - Σ φ_i w_{t-i} - Σ θ_j e_{t-j}`` with zero
pre-sample values, and minimizes the SSE (conditioning on the first
``max(p, q)`` terms) with L-BFGS-B. Forecast standard errors come from the
psi-weight expansion of the d-times-integrated ARMA process. ``AutoARIMA``
picks ``d`` with a variance heuristic and ``(p, q)`` by AICc grid search.

The constant ``c`` (``include_mean``) is estimated only when ``d == 0``; for
differenced models (``d > 0``) ``include_mean`` is silently ignored, mirroring
R's ``arima`` ``include.mean`` semantics — there is no drift term.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, signal

from ..core.base import PerSeriesForecaster
from ..core.exceptions import ForecastOSError
from ..core.registry import register

_COEF_BOUND = 0.99


def _css_residuals(w: np.ndarray, c: float, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Innovations of the CSS recursion, with pre-sample w and e taken as 0."""
    u = w - c
    for i, ph in enumerate(phi, start=1):
        u[i:] -= ph * w[:-i]
    if len(theta):
        u = signal.lfilter([1.0], np.concatenate([[1.0], theta]), u)
    return u


def _fit_css(y: np.ndarray, order: tuple[int, int, int], include_mean: bool) -> dict:
    """Fit one series by CSS; returns the per-series state dict."""
    p, d, q = order
    w = np.diff(y, n=d) if d else np.asarray(y, dtype=float)
    lasts = [float(np.diff(y, n=i)[-1]) for i in range(d)]
    m = max(p, q)
    use_c = include_mean and d == 0

    def unpack(params: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        i = 1 if use_c else 0
        c = float(params[0]) if use_c else 0.0
        return c, np.asarray(params[i : i + p]), np.asarray(params[i + p : i + p + q])

    def objective(params: np.ndarray) -> float:
        c, phi, theta = unpack(params)
        e = _css_residuals(w, c, phi, theta)
        return float(np.sum(e[m:] ** 2))

    x0: list[float] = [float(np.mean(w))] if use_c else []
    bounds: list[tuple[float | None, float | None]] = [(None, None)] if use_c else []
    x0 += [0.0] * (p + q)
    bounds += [(-_COEF_BOUND, _COEF_BOUND)] * (p + q)

    if x0:
        res = optimize.minimize(objective, np.asarray(x0), method="L-BFGS-B", bounds=bounds)
        c, phi, theta = unpack(res.x)
        converged = bool(res.success)
    else:
        c, phi, theta = 0.0, np.zeros(0), np.zeros(0)
        converged = True

    e = _css_residuals(w, c, phi, theta)
    fitted = np.full(len(y), np.nan)
    fitted[d + m :] = y[d + m :] - e[m:]
    return {
        "fitted": fitted,
        "c": c,
        "phi": phi,
        "theta": theta,
        "w": w,
        "e": e,
        "lasts": lasts,
        "d_": d,
        "sse": float(np.sum(e[m:] ** 2)),
        "n_eff": len(w) - m,
        "converged": converged,
    }


def _forecast_css(state: dict, h: int) -> np.ndarray:
    """Iterate the recursion with future e=0, then invert the differencing."""
    c, phi, theta = state["c"], state["phi"], state["theta"]
    w, e = state["w"], state["e"]
    p, q = len(phi), len(theta)
    n = len(w)
    wext = np.concatenate([w, np.zeros(h)])
    fw = np.empty(h)
    for k in range(h):
        t = n + k
        acc = c
        for i in range(1, p + 1):
            if t - i >= 0:
                acc += phi[i - 1] * wext[t - i]
        for j in range(1, q + 1):
            if 0 <= t - j < n:
                acc += theta[j - 1] * e[t - j]
        fw[k] = acc
        wext[t] = acc
    for last in reversed(state["lasts"]):
        fw = last + np.cumsum(fw)
    return fw


def _psi_sigma(state: dict, h: int) -> np.ndarray:
    """Forecast std errors from psi weights, cumulatively summed d times."""
    phi, theta, d = state["phi"], state["theta"], state["d_"]
    p, q = len(phi), len(theta)
    psi = np.zeros(h)
    psi[0] = 1.0
    for j in range(1, h):
        val = theta[j - 1] if j <= q else 0.0
        for i in range(1, min(j, p) + 1):
            val += phi[i - 1] * psi[j - i]
        psi[j] = val
    for _ in range(d):
        psi = np.cumsum(psi)
    return state["_sigma"] * np.sqrt(np.cumsum(psi**2))


@register("arima", family="statistical")
class ARIMA(PerSeriesForecaster):
    """ARIMA(p, d, q) forecaster estimated by conditional sum of squares.

    ``include_mean`` applies only when ``d == 0``: for differenced models
    (``d > 0``) it is silently ignored, mirroring R's ``arima``
    ``include.mean`` semantics, so e.g. ARIMA(0, 1, 0) forecasts are flat at
    the last observation with no drift term. :class:`AutoARIMA` can still
    capture drift-like behavior through AR structure on the differenced
    series.
    """

    def __init__(self, order: tuple[int, int, int] = (1, 1, 1), include_mean: bool = True):
        try:
            order = tuple(int(v) for v in order)
        except (TypeError, ValueError) as exc:
            raise ForecastOSError(f"order must be three ints (p, d, q), got {order!r}") from exc
        if len(order) != 3 or any(v < 0 for v in order):
            raise ForecastOSError(f"order must be three non-negative ints (p, d, q), got {order!r}")
        self.order = order
        self.include_mean = include_mean
        p, d, q = order
        self.min_train_size = max(p, q) + d + 5

    def _fit_series(self, y: np.ndarray) -> dict:
        return _fit_css(y, self.order, self.include_mean)

    def _predict_series(self, state: dict, h: int) -> np.ndarray:
        return _forecast_css(state, h)

    def _predict_sigma(self, state: dict, h: int) -> np.ndarray:
        return _psi_sigma(state, h)


@register("auto_arima", family="statistical")
class AutoARIMA(PerSeriesForecaster):
    """Automatic ARIMA: variance-heuristic d selection + AICc grid over (p, q)."""

    def __init__(self, max_p: int = 3, max_d: int = 2, max_q: int = 3):
        self.max_p = int(max_p)
        self.max_d = int(max_d)
        self.max_q = int(max_q)
        if min(self.max_p, self.max_d, self.max_q) < 0:
            raise ForecastOSError("max_p, max_d, and max_q must be non-negative")
        self.min_train_size = max(self.max_p, self.max_q) + self.max_d + 5

    def _select_d(self, y: np.ndarray) -> int:
        """d = argmin of std(diff^d(y)) * 1.05^d (penalizes over-differencing)."""
        best_d, best_score = 0, np.inf
        for d in range(self.max_d + 1):
            w = np.diff(y, n=d) if d else y
            if len(w) < 3:
                break
            score = float(np.std(w)) * 1.05**d
            if score < best_score:
                best_d, best_score = d, score
        return best_d

    def _fit_series(self, y: np.ndarray) -> dict:
        d = self._select_d(y)
        best_state, best_aicc, best_order = None, np.inf, None
        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                state = _fit_css(y, (p, d, q), include_mean=True)
                if not state["converged"]:
                    continue
                n, k = state["n_eff"], p + q + 1
                if n - k - 1 <= 0:
                    continue
                aicc = n * np.log(max(state["sse"] / n, 1e-12)) + 2 * k
                aicc += 2 * k * (k + 1) / (n - k - 1)
                if aicc < best_aicc:
                    best_state, best_aicc, best_order = state, aicc, (p, d, q)
        if best_state is None:  # every candidate failed to converge
            best_state, best_order = _fit_css(y, (0, d, 0), include_mean=True), (0, d, 0)
        best_state["order_"] = best_order
        return best_state

    def _predict_series(self, state: dict, h: int) -> np.ndarray:
        return _forecast_css(state, h)

    def _predict_sigma(self, state: dict, h: int) -> np.ndarray:
        return _psi_sigma(state, h)
