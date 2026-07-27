"""GARCH(1,1) volatility modeling.

``GARCH11`` is a standalone estimator for a single returns array: conditional
maximum likelihood of ``sigma2_t = omega + alpha*r_{t-1}^2 + beta*sigma2_{t-1}``
(returns demeaned and z-scored internally for a well-conditioned optimization,
parameters reported back on the original scale). ``GARCHVolatility`` wraps it
as a registered forecaster whose ``yhat`` is the predicted per-period
conditional volatility of each series.

The likelihood is optimized in ``(lam, p, q)`` coordinates — long-run variance,
persistence ``alpha + beta``, and the share of that persistence carried by
``alpha`` — with an exact analytic gradient. Those coordinates decorrelate
``omega`` from the persistence and turn the ``alpha + beta < 1`` stationarity
constraint into a plain box bound, which is what makes the optimization
converge; ``fit`` raises :class:`~forecast_os.core.exceptions.ForecastOSError`
rather than returning parameters from a run the optimizer never converged.
The search is a multistart over :data:`_STARTS`, so the reported parameters are
the best converged stationary point found — not a certified global optimum.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..core.base import PerSeriesForecaster
from ..core.exceptions import ForecastOSError, NotFittedError
from ..core.registry import register

__all__ = ["GARCH11", "GARCHVolatility"]

_MIN_OBS = 10
_MAX_PERSISTENCE = 0.999
# (persistence, alpha share) starting points; omega starts variance-targeted.
# The grid spans low as well as high persistence: a grid confined to p >= 0.90
# never puts a start in the right basin for series whose optimum is weakly
# persistent (iid returns), and missed it by up to 0.55 nats.
_STARTS = tuple(
    (p, q) for p in (0.10, 0.30, 0.60, 0.90, 0.95, 0.98) for q in (0.05, 0.15, 0.35, 0.70)
)


def _sigma2_path(r2: np.ndarray, omega: float, alpha: float, beta: float, var0: float):
    sigma2 = np.empty(r2.size)
    sigma2[0] = var0
    for t in range(1, r2.size):
        sigma2[t] = omega + alpha * r2[t - 1] + beta * sigma2[t - 1]
    return np.maximum(sigma2, 1e-12)


def _nll_grad(params: np.ndarray, r2: np.ndarray, var0: float):
    """Conditional Gaussian negative log-likelihood and its exact gradient.

    The likelihood is ``0.5 * sum(log sigma2_t + r2_t / sigma2_t)`` (dropping
    the constant term) and the gradient is wrt ``(omega, alpha, beta)``.
    Stationarity (``alpha + beta <= _MAX_PERSISTENCE``) is imposed by the
    optimizer's box bounds in :func:`_objective` coordinates, not by a penalty.

    The variance recursion is differentiated alongside itself: each
    ``d sigma2_t / d theta`` obeys the same AR(1) recursion in ``beta``, so one
    pass yields the likelihood and all three partials. A finite-difference
    gradient on this objective is noisy enough to abort L-BFGS-B's line search.
    """
    omega, alpha, beta = params
    n = r2.size
    sigma2 = np.empty(n)
    d_omega = np.zeros(n)
    d_alpha = np.zeros(n)
    d_beta = np.zeros(n)
    sigma2[0] = var0
    for t in range(1, n):
        sigma2[t] = omega + alpha * r2[t - 1] + beta * sigma2[t - 1]
        d_omega[t] = 1.0 + beta * d_omega[t - 1]
        d_alpha[t] = r2[t - 1] + beta * d_alpha[t - 1]
        d_beta[t] = sigma2[t - 1] + beta * d_beta[t - 1]
    sigma2 = np.maximum(sigma2, 1e-12)
    nll = 0.5 * float(np.sum(np.log(sigma2) + r2 / sigma2))
    w = 0.5 * (1.0 / sigma2 - r2 / sigma2**2)
    grad = np.array([float(w @ d_omega), float(w @ d_alpha), float(w @ d_beta)])
    return nll, grad


def _unpack(theta: np.ndarray) -> tuple[float, float, float]:
    """``(lam, p, q)`` -> ``(omega, alpha, beta)``."""
    lam, p, q = theta
    return lam * (1.0 - p), p * q, p * (1.0 - q)


def _objective(theta: np.ndarray, r2: np.ndarray, var0: float):
    """:func:`_nll_grad` in the well-conditioned ``(lam, p, q)`` coordinates."""
    lam, p, q = theta
    nll, (g_omega, g_alpha, g_beta) = _nll_grad(np.asarray(_unpack(theta)), r2, var0)
    grad = np.array(
        [
            g_omega * (1.0 - p),
            -g_omega * lam + g_alpha * q + g_beta * (1.0 - q),
            p * (g_alpha - g_beta),
        ]
    )
    return nll, grad


class GARCH11:
    """GARCH(1,1) conditional-volatility model estimated by MLE.

    After :meth:`fit`, ``converged_`` and ``opt_message_`` report the optimizer
    verdict for the run whose parameters were kept. ``converged_`` is ``True``
    when a gradient (L-BFGS-B) start converged and its parameters were kept, and
    ``False`` when every gradient start stalled and the returned parameters come
    from the gradient-free retry — a weaker fit, worth screening out of a panel.
    :meth:`fit` raises when not even that retry converges.

    What is guaranteed is a converged stationary point of the likelihood, the
    best one a multistart over :data:`_STARTS` found; global optimality is not
    guaranteed and should not be assumed downstream.
    """

    def fit(self, returns: np.ndarray) -> GARCH11:
        """Estimate (omega, alpha, beta) on ``returns`` (demeaned internally).

        Keeps the best converged run of a multistart over :data:`_STARTS`, and
        raises :class:`~forecast_os.core.exceptions.ForecastOSError` when the
        likelihood optimization does not converge from any starting point.
        """
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
        # lam is the long-run variance omega / (1 - p), so the historical
        # ceiling omega <= 10*var0 is the bound below. Reusing 10*var0 verbatim
        # here would instead assert "long-run variance <= 10x sample variance",
        # a modelling restriction that binds on near-integrated series.
        bounds = [
            (1e-8, 10.0 * var0 / (1.0 - _MAX_PERSISTENCE)),
            (0.0, _MAX_PERSISTENCE),
            (0.0, 1.0),
        ]
        best = None  # best *converged* run
        fallback = None  # best run of any kind, converged or not
        for p0, q0 in _STARTS:
            res = minimize(
                _objective,
                np.array([var0, p0, q0]),
                args=(z2, var0),
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
            )
            if fallback is None or res.fun < fallback.fun:
                fallback = res
            if res.success and (best is None or res.fun < best.fun):
                best = res
        gradient_converged = best is not None
        if best is None:
            # Every gradient run stalled: polish the best point gradient-free
            # before giving up, so a merely awkward series still fits.
            best = minimize(
                lambda theta: _objective(theta, z2, var0)[0],
                fallback.x,
                method="Nelder-Mead",
                bounds=bounds,
                options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 4000},
            )
            if not best.success or best.fun > fallback.fun:
                raise ForecastOSError(
                    "GARCH(1,1) maximum likelihood did not converge from any of "
                    f"{len(_STARTS)} starting points (best objective "
                    f"{fallback.fun!r}); optimizer reported {fallback.message!r}. "
                    "Refusing to return parameters from a run that never "
                    "converged — the returns may be too short or too irregular "
                    "for GARCH(1,1)."
                )
        self.converged_ = gradient_converged
        self.opt_message_ = str(best.message)
        omega_z, alpha, beta = _unpack(best.x)
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
