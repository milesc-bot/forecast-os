"""Kalman-filter state-space forecasters (local level / local linear trend).

Noise variances (observation R, diagonal state Q) are estimated by maximizing
the Gaussian likelihood of the prediction-error decomposition. Forecasts
propagate the filtered state ``h`` steps and expose native standard errors
``sqrt(H P_k H' + R)``, so prediction intervals widen with the horizon.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from ..core.base import PerSeriesForecaster
from ..core.exceptions import ForecastOSError
from ..core.registry import register

_MODELS = ("local_level", "local_linear")


def _system(model: str) -> tuple[np.ndarray, np.ndarray]:
    """Transition matrix F and observation vector H for the named model."""
    if model == "local_level":
        return np.array([[1.0]]), np.array([1.0])
    return np.array([[1.0, 1.0], [0.0, 1.0]]), np.array([1.0, 0.0])


def _series_scale(y: np.ndarray) -> float:
    """Positive factor that carries the units of ``y``, used to fit at unit scale.

    The Gaussian likelihood of a linear state-space model satisfies
    ``nll(c*y; c**2 R, c**2 Q, c**2 P0) = nll(y; R, Q, P0) + (n - k)*log(c)``
    (each retained prediction error contributes ``0.5*log(S)`` and :func:`_filter`
    skips the first ``k`` of the ``n``), so the MLE is exactly equivariant in the
    units of ``y`` — the offset does not depend on the parameters, so the argmin
    is unchanged. The optimizer, though, searches *absolute* log-variances inside
    a fixed box, which is not. Fitting ``y / scale`` and multiplying the variances
    back by ``scale**2`` leaves the optimum
    untouched — variances carry the square of the units, the state and the
    one-step predictions carry them linearly — while giving the optimizer a
    well-conditioned O(1) problem at any magnitude.
    """
    s = float(np.std(np.diff(y)))
    if not np.isfinite(s) or s <= 0.0:  # constant first differences: use the level
        s = float(np.std(y))
    if not np.isfinite(s) or s <= 0.0:  # exactly constant series
        s = float(np.max(np.abs(y)))
    return s if np.isfinite(s) and s > 0.0 else 1.0


def _filter(
    y: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    r: float,
    qdiag: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """One filter pass: (nll, one-step predictions, prior x_T+1, prior P_T+1).

    The negative log-likelihood skips the first ``len(x0)`` prediction errors,
    which are dominated by the diffuse prior.
    """
    n = len(y)
    k = len(x0)
    Q = np.diag(qdiag)
    x, P = x0.astype(float).copy(), P0.copy()
    preds = np.empty(n)
    nll = 0.0
    for t in range(n):
        yhat = float(H @ x)
        S = float(H @ P @ H) + r
        v = y[t] - yhat
        preds[t] = yhat
        if t >= k:
            nll += 0.5 * (np.log(S) + v * v / S)
        K = (P @ H) / S
        x = x + K * v
        P = P - np.outer(K, H @ P)
        x = F @ x
        P = F @ P @ F.T + Q
        P = 0.5 * (P + P.T)
    return nll, preds, x, P


@register("kalman", family="statistical")
class KalmanForecaster(PerSeriesForecaster):
    """Kalman filter forecaster with MLE noise variances and growing intervals."""

    min_train_size = 5

    def __init__(self, model: str = "local_level"):
        if model not in _MODELS:
            raise ForecastOSError(
                f"unknown state-space model {model!r}; expected one of {_MODELS}"
            )
        self.model = model

    def _fit_series(self, y: np.ndarray) -> dict:
        F, H = _system(self.model)
        k = F.shape[0]
        # Fit a unit-scale copy: the (-30, 30) log-variance box is then a box on
        # the variance *ratios* to var(diff(y)), which is scale-free, instead of
        # clamping R and Q on large- or small-magnitude series.
        scale = _series_scale(y)
        z = y / scale
        base_var = max(float(np.var(np.diff(z))), 1e-8)
        if self.model == "local_level":
            x0 = np.array([z[0]])
        else:
            x0 = np.array([z[0], float(np.mean(np.diff(z[:10])))])
        p0 = 1e4 * float(np.var(z))
        P0 = (p0 if p0 > 0 else 1.0) * np.eye(k)

        def nll(logparams: np.ndarray) -> float:
            return _filter(z, F, H, float(np.exp(logparams[0])), np.exp(logparams[1:]), x0, P0)[0]

        init = np.full(1 + k, np.log(base_var))
        res = optimize.minimize(nll, init, method="L-BFGS-B", bounds=[(-30.0, 30.0)] * (1 + k))
        r = float(np.exp(res.x[0]))
        qdiag = np.exp(res.x[1:])
        _, preds, x_last, P_last = _filter(z, F, H, r, qdiag, x0, P0)
        # Back to the units of y: states are linear in the scale, variances square.
        fitted = preds * scale
        fitted[:k] = np.nan  # diffuse-prior warm-up
        # ``scale * scale`` rather than ``scale ** 2``: float pow raises a bare
        # OverflowError past ~1.3e154, while the product saturates to inf like
        # every other quantity in the filter.
        scale_sq = scale * scale
        return {"fitted": fitted, "x_": x_last * scale, "P_": P_last * scale_sq,
                "F_": F, "H_": H, "Q_": np.diag(qdiag) * scale_sq, "R_": r * scale_sq}

    def _predict_series(self, state: dict, h: int) -> np.ndarray:
        F, H = state["F_"], state["H_"]
        x = state["x_"].copy()
        out = np.empty(h)
        for i in range(h):
            out[i] = float(H @ x)
            x = F @ x
        return out

    def _predict_sigma(self, state: dict, h: int) -> np.ndarray:
        F, H, Q, r = state["F_"], state["H_"], state["Q_"], state["R_"]
        P = state["P_"].copy()
        sig = np.empty(h)
        for i in range(h):
            sig[i] = np.sqrt(float(H @ P @ H) + r)
            P = F @ P @ F.T + Q
        return sig
