"""Seasonal ARIMA (SARIMA) estimated by conditional sum of squares.

``SARIMA(p, d, q)(P, D, Q)_m`` models the multiplicative seasonal process

    phi(B) Phi(B^m) (1 - B)^d (1 - B^m)^D y_t = c + theta(B) Theta(B^m) e_t

Estimation reuses the ARIMA CSS machinery: the multiplicative polynomials are
expanded into equivalent long AR/MA polynomials of orders ``p + P*m`` and
``q + Q*m``, and the ordinary innovation recursion of :mod:`.arima` is run on
the doubly differenced series. Only the ``p + q + P + Q`` structural
coefficients (plus an optional constant) are optimized — the expansion is
recomputed inside the objective, so the multiplicative constraint is enforced
exactly rather than approximated by a long unconstrained polynomial.

Sign conventions follow :mod:`.arima` (Hamilton, ``+`` on the MA terms)::

    AR:  A(B) = 1 - sum_i phi_i B^i          MA:  M(B) = 1 + sum_j theta_j B^j

so the seasonal factors are ``1 - sum_k Phi_k B^{k*m}`` and
``1 + sum_k Theta_k B^{k*m}``.

Differencing is applied seasonally first and then regularly,
``w = grad^d (grad_m^D y)``, keeping the partially differenced series at every
stage so forecasts can be integrated back in reverse order. Forecast standard
errors come from the psi weights of the *expanded* ARMA polynomial pushed
through the same integration operator.

The constant ``c`` (``include_mean``) is estimated only when ``d == 0`` **and**
``D == 0``; any differencing silently drops it, mirroring the ``ARIMA``
semantics (there is no drift term).
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from ..core.base import PerSeriesForecaster
from ..core.exceptions import ForecastOSError
from ..core.registry import register
from .arima import _COEF_BOUND, _css_residuals, _css_scale, _fit_css, _forecast_css

#: Seasonal strength above which :class:`AutoSARIMA` takes a seasonal
#: difference (the usual ``nsdiffs`` cut-off).
_SEASONAL_STRENGTH_THRESHOLD = 0.64

__all__ = ["SARIMA", "AutoSARIMA"]


# -- polynomial expansion -----------------------------------------------------


def _lag_poly(coefs: np.ndarray, step: int, sign: float) -> np.ndarray:
    """Dense coefficients of ``1 + sign * sum_k coefs[k] B^{k*step}``."""
    coefs = np.asarray(coefs, dtype=float)
    poly = np.zeros(len(coefs) * step + 1)
    poly[0] = 1.0
    for k, coef in enumerate(coefs, start=1):
        poly[k * step] = sign * coef
    return poly


def _expand_ar(phi: np.ndarray, seasonal_phi: np.ndarray, m: int) -> np.ndarray:
    """AR coefficients of ``phi(B) Phi(B^m)`` in the ``1 - sum phi_i B^i`` form.

    Convolves ``[1, -phi_1, ..., -phi_p]`` with the seasonal polynomial
    ``[1, 0, ..., -Phi_1, ...]`` and flips the sign back, so the result plugs
    straight into the ``+phi`` recursion of :func:`~.arima._css_residuals`.
    """
    ar_poly = np.convolve(_lag_poly(phi, 1, -1.0), _lag_poly(seasonal_phi, m, -1.0))
    return -ar_poly[1:]


def _expand_ma(theta: np.ndarray, seasonal_theta: np.ndarray, m: int) -> np.ndarray:
    """MA coefficients of ``theta(B) Theta(B^m)`` in the ``1 + sum theta_j B^j`` form."""
    ma_poly = np.convolve(_lag_poly(theta, 1, 1.0), _lag_poly(seasonal_theta, m, 1.0))
    return ma_poly[1:]


# -- differencing / integration -----------------------------------------------


def _seasonal_difference(
    y: np.ndarray, d: int, D: int, m: int
) -> tuple[np.ndarray, list[tuple[str, object]]]:
    """Apply ``D`` seasonal then ``d`` regular differences to ``y``.

    Returns ``(w, stages)`` where ``w = grad^d (grad_m^D y)`` and ``stages``
    records, in application order, what :func:`_seasonal_integrate` needs to
    undo each step: the last value before a regular difference, or the last
    ``m`` values before a seasonal one.
    """
    cur = np.asarray(y, dtype=float)
    stages: list[tuple[str, object]] = []
    for _ in range(D):
        if len(cur) <= m:
            raise ForecastOSError(
                f"series is too short for {D} seasonal difference(s) at m={m}: "
                f"only {len(cur)} observation(s) left before differencing"
            )
        stages.append(("seasonal", cur[-m:].copy()))
        cur = cur[m:] - cur[:-m]
    for _ in range(d):
        if len(cur) < 2:
            raise ForecastOSError(
                f"series is too short for {d} regular difference(s): only "
                f"{len(cur)} observation(s) left before differencing"
            )
        stages.append(("regular", float(cur[-1])))
        cur = np.diff(cur)
    return cur, stages


def _seasonal_integrate(fw: np.ndarray, stages: list[tuple[str, object]], m: int) -> np.ndarray:
    """Undo :func:`_seasonal_difference` on a forecast path, latest stage first.

    A regular stage integrates by ``f_prev = last + cumsum(f)``. A seasonal
    stage resolves ``f_prev[i] = stage[n + i - m] + f[i]`` recursively: the
    first ``m`` steps read the stored tail of the pre-difference series, later
    steps read the values this stage has already produced.
    """
    out = np.asarray(fw, dtype=float)
    for kind, info in reversed(stages):
        if kind == "regular":
            out = info + np.cumsum(out)
        else:
            tail = np.asarray(info, dtype=float)
            res = np.empty(len(out))
            for i in range(len(out)):
                prev = tail[i] if i < m else res[i - m]
                res[i] = prev + out[i]
            out = res
    return out


# -- psi weights --------------------------------------------------------------


def _sarima_psi(
    phi: np.ndarray, theta: np.ndarray, h: int, d: int, D: int, m: int
) -> np.ndarray:
    """Psi weights of the expanded ARMA process, integrated ``d``/``D`` times.

    Regular integration is a cumulative sum; each seasonal integration adds
    ``psi[j - m]`` into ``psi[j]`` ascending in ``j`` (multiplication by
    ``1 / (1 - B^m)``). The two operators commute, so the order is immaterial.
    """
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
    for _ in range(D):
        for j in range(m, h):
            psi[j] += psi[j - m]
    return psi


def _sarima_sigma(state: dict, h: int) -> np.ndarray:
    """Forecast standard errors from the integrated psi weights."""
    psi = _sarima_psi(
        state["phi"], state["theta"], h, state["d_"], state["D_"], state["m_"]
    )
    return state["_sigma"] * np.sqrt(np.cumsum(psi**2))


# -- estimation ---------------------------------------------------------------


def _fit_sarima_css(
    y: np.ndarray,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int],
    m: int,
    include_mean: bool,
) -> dict:
    """Fit one series by CSS on the expanded polynomials; returns the state dict.

    ``phi``/``theta`` in the returned state are the EXPANDED recursion
    coefficients (so :func:`~.arima._forecast_css` can be reused verbatim);
    the structural coefficients are kept alongside as ``phi_raw``,
    ``theta_raw``, ``seasonal_phi`` and ``seasonal_theta``. ``lasts`` is
    deliberately empty — integration is driven by ``stages`` instead, and
    ``_forecast_css`` therefore returns the forecast on the differenced scale.
    """
    p, d, q = order
    P, D, Q = seasonal_order
    w, stages = _seasonal_difference(y, d, D, m)
    n_warm = max(p + P * m, q + Q * m)
    use_c = include_mean and d == 0 and D == 0

    if P == 0 and Q == 0 and D == 0:
        # No seasonal structure survives: this is exactly ARIMA(p, d, q), so
        # delegate to the shared implementation and stay bit-identical to it.
        base = _fit_css(y, order, include_mean)
        c, phi, theta = base["c"], base["phi"], base["theta"]
        seasonal_phi = seasonal_theta = np.zeros(0)
        e, fitted, converged = base["e"], base["fitted"], base["converged"]
        phi_e, theta_e = phi, theta
    else:

        def unpack(
            params: np.ndarray,
        ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            """Split the flat parameter vector into (c, phi, theta, Phi, Theta)."""
            i = 1 if use_c else 0
            c = float(params[0]) if use_c else 0.0
            phi = np.asarray(params[i : i + p])
            theta = np.asarray(params[i + p : i + p + q])
            j = i + p + q
            return (
                c,
                phi,
                theta,
                np.asarray(params[j : j + P]),
                np.asarray(params[j + P : j + P + Q]),
            )

        # Fit on a unit-scale copy: L-BFGS-B's gradient tolerance is absolute
        # while the CSS gradient scales as the units of y squared. See
        # :func:`~.arima._css_scale`.
        scale = _css_scale(w)
        w_fit = w / scale

        def objective(params: np.ndarray) -> float:
            c, phi, theta, sphi, stheta = unpack(params)
            e = _css_residuals(w_fit, c, _expand_ar(phi, sphi, m), _expand_ma(theta, stheta, m))
            return float(np.sum(e[n_warm:] ** 2))

        n_coef = p + q + P + Q
        x0: list[float] = [float(np.mean(w_fit))] if use_c else []
        bounds: list[tuple[float | None, float | None]] = [(None, None)] if use_c else []
        x0 += [0.0] * n_coef
        bounds += [(-_COEF_BOUND, _COEF_BOUND)] * n_coef

        if x0:
            res = optimize.minimize(objective, np.asarray(x0), method="L-BFGS-B", bounds=bounds)
            c, phi, theta, seasonal_phi, seasonal_theta = unpack(res.x)
            c *= scale  # the intercept carries the units; the coefficients do not
            converged = bool(res.success)
        else:  # nothing to estimate: (0,d,0)(0,D,0) with no constant
            c = 0.0
            phi = theta = seasonal_phi = seasonal_theta = np.zeros(0)
            converged = True

        phi_e = _expand_ar(phi, seasonal_phi, m)
        theta_e = _expand_ma(theta, seasonal_theta, m)
        e = _css_residuals(w, c, phi_e, theta_e)
        off = d + D * m
        fitted = np.full(len(y), np.nan)
        fitted[off + n_warm :] = y[off + n_warm :] - e[n_warm:]

    return {
        "fitted": fitted,
        "c": c,
        "phi": phi_e,
        "theta": theta_e,
        "phi_raw": phi,
        "theta_raw": theta,
        "seasonal_phi": seasonal_phi,
        "seasonal_theta": seasonal_theta,
        "w": w,
        "e": e,
        "lasts": [],
        "stages": stages,
        "d_": d,
        "D_": D,
        "m_": m,
        "order_": (p, d, q),
        "seasonal_order_": (P, D, Q),
        "sse": float(np.sum(e[n_warm:] ** 2)),
        "n_eff": len(w) - n_warm,
        "converged": converged,
    }


def _forecast_sarima(state: dict, h: int) -> np.ndarray:
    """Iterate the expanded recursion, then undo the seasonal/regular differencing."""
    return _seasonal_integrate(_forecast_css(state, h), state["stages"], state["m_"])


# -- order selection helpers --------------------------------------------------


def _aicc(n: int, sse: float, k: int) -> float:
    """CSS AICc, identical to :class:`~.arima.AutoARIMA`'s.

    ``n·log(sse/n) + 2k + 2k(k+1)/(n-k-1)`` with ``k`` the number of estimated
    coefficients plus one (for the innovation variance). Callers must ensure
    ``n - k - 1 > 0``.
    """
    return float(n * np.log(max(sse / n, 1e-12)) + 2 * k + 2 * k * (k + 1) / (n - k - 1))


def _seasonal_caps(m: int, n_after: int, max_P: int, max_Q: int) -> tuple[int, int]:
    """Cap the seasonal grid to ``(0, 0)`` when seasonal terms are unidentifiable.

    At ``m == 1`` the seasonal lags coincide with the regular ones, and a
    series with fewer than two full seasons left after differencing cannot
    identify a lag-``m`` coefficient; in both cases the seasonal dimensions of
    the grid are dropped, which also keeps the search fast on short series.
    """
    if m > 1 and n_after >= 2 * m + 5:
        return max_P, max_Q
    return 0, 0


# -- argument validation ------------------------------------------------------


def _check_order(order, label: str, letters: str) -> tuple[int, int, int]:
    try:
        order = tuple(int(v) for v in order)
    except (TypeError, ValueError) as exc:
        raise ForecastOSError(
            f"{label} must be three ints ({letters}), got {order!r}"
        ) from exc
    if len(order) != 3 or any(v < 0 for v in order):
        raise ForecastOSError(
            f"{label} must be three non-negative ints ({letters}), got {order!r}"
        )
    return order


def _check_m(m) -> int:
    try:
        m_int = int(m)
    except (TypeError, ValueError) as exc:
        raise ForecastOSError(f"m must be an integer >= 1, got {m!r}") from exc
    if m_int < 1:
        raise ForecastOSError(f"m must be an integer >= 1, got {m!r}")
    return m_int


# -- models -------------------------------------------------------------------


@register("sarima", family="statistical")
class SARIMA(PerSeriesForecaster):
    """SARIMA(p, d, q)(P, D, Q)_m estimated by conditional sum of squares.

    The multiplicative seasonal polynomials are expanded into long AR/MA
    polynomials of orders ``p + P*m`` and ``q + Q*m`` and driven through the
    ARIMA CSS recursion, so the sign conventions match :class:`~.arima.ARIMA`
    exactly: ``AR = 1 - sum phi_i B^i``, ``MA = 1 + sum theta_j B^j``.

    ``include_mean`` applies only when ``d == 0`` and ``D == 0``; any
    differencing silently drops the constant, mirroring the ``ARIMA``
    semantics. With ``seasonal_order=(0, 0, 0)`` the fit is numerically
    identical to ``ARIMA(order)``.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 0, 0),
        seasonal_order: tuple[int, int, int] = (1, 1, 0),
        m: int = 12,
        include_mean: bool = True,
    ):
        self.order = _check_order(order, "order", "p, d, q")
        self.seasonal_order = _check_order(seasonal_order, "seasonal_order", "P, D, Q")
        self.m = _check_m(m)
        self.include_mean = include_mean
        p, d, q = self.order
        P, D, Q = self.seasonal_order
        self.min_train_size = max(p + P * self.m, q + Q * self.m) + d + D * self.m + 5

    def _fit_series(self, y: np.ndarray) -> dict:
        return _fit_sarima_css(
            y, self.order, self.seasonal_order, self.m, self.include_mean
        )

    def _predict_series(self, state: dict, h: int) -> np.ndarray:
        return _forecast_sarima(state, h)

    def _predict_sigma(self, state: dict, h: int) -> np.ndarray:
        return _sarima_sigma(state, h)


@register("auto_sarima", family="statistical")
class AutoSARIMA(PerSeriesForecaster):
    """Automatic SARIMA: seasonal-strength D, variance-heuristic d, AICc (p, q, P, Q).

    ``D`` is taken while the seasonal strength ``1 - Var(remainder) /
    Var(detrended)`` of a classical decomposition exceeds 0.64; ``d`` then uses
    the same variance heuristic as :class:`~.arima.AutoARIMA` on the
    seasonally differenced series. The remaining orders are chosen by AICc,
    ``n·log(sse/n) + 2k + 2k(k+1)/(n-k-1)`` with ``k = p + q + P + Q + 1``.

    The seasonal part of the grid is disabled for ``m == 1`` (where it would
    duplicate the regular lags) and for series too short to identify seasonal
    coefficients, keeping the search fast on small inputs.
    """

    def __init__(
        self,
        m: int = 12,
        max_p: int = 2,
        max_q: int = 2,
        max_P: int = 1,
        max_Q: int = 1,
        max_d: int = 1,
        max_D: int = 1,
    ):
        self.m = _check_m(m)
        limits = {
            "max_p": max_p, "max_q": max_q, "max_P": max_P,
            "max_Q": max_Q, "max_d": max_d, "max_D": max_D,
        }
        for name, value in limits.items():
            try:
                ivalue = int(value)
            except (TypeError, ValueError) as exc:
                raise ForecastOSError(
                    f"{name} must be a non-negative int, got {value!r}"
                ) from exc
            if ivalue < 0:
                raise ForecastOSError(f"{name} must be a non-negative int, got {value!r}")
            setattr(self, name, ivalue)
        self.min_train_size = (
            max(self.max_p + self.max_P * self.m, self.max_q + self.max_Q * self.m)
            + self.max_d
            + self.max_D * self.m
            + 5
        )

    def _select_D(self, y: np.ndarray) -> int:
        """Seasonal differences to take, by repeated seasonal-strength testing."""
        if self.m < 2 or self.max_D < 1:
            return 0
        cur, D = np.asarray(y, dtype=float), 0
        while D < self.max_D and len(cur) >= 3 * self.m:
            if _seasonal_strength(cur, self.m) <= _SEASONAL_STRENGTH_THRESHOLD:
                break
            cur = cur[self.m :] - cur[: -self.m]
            D += 1
        return D

    def _select_d(self, y: np.ndarray) -> int:
        """d = argmin of std(diff^d(y)) * 1.05^d, as in :class:`~.arima.AutoARIMA`."""
        best_d, best_score = 0, np.inf
        for d in range(self.max_d + 1):
            w = np.diff(y, n=d) if d else y
            if len(w) < 3:
                break
            score = float(np.std(w)) * 1.05**d
            if score < best_score:
                best_d, best_score = d, score
        return best_d

    def _search_orders(self, y: np.ndarray, d: int, D: int) -> dict:
        """AICc grid over (p, q, P, Q) at the chosen differencing."""
        m = self.m
        n_after = len(y) - d - D * m
        max_P, max_Q = _seasonal_caps(m, n_after, self.max_P, self.max_Q)
        # Every candidate must be scored on the SAME observations. Each model's own
        # warm-up (max(p + P*m, q + Q*m)) differs by up to ~2*m across the grid, and
        # `n*log(sse/n)` shifts by 2*n*log(lambda) when y is rescaled — so a
        # per-candidate n makes AICc differences depend on the units of y, and a
        # change of units silently flips the selected order. A common window makes
        # the comparison scale invariant.
        nw = max(self.max_p + max_P * m, self.max_q + max_Q * m)
        best_state, best_aicc = None, np.inf
        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                for P in range(max_P + 1):
                    for Q in range(max_Q + 1):
                        n_warm = max(p + P * m, q + Q * m)
                        if n_after - n_warm < 5:
                            continue
                        state = _fit_sarima_css(y, (p, d, q), (P, D, Q), m, True)
                        if not state["converged"]:
                            continue
                        e = np.asarray(state["e"], dtype=float)
                        n, k = len(e) - nw, p + q + P + Q + 1
                        if n - k - 1 <= 0:
                            continue
                        sse = float(np.sum(e[nw:] ** 2))
                        aicc = _aicc(n, sse, k)
                        if aicc < best_aicc:
                            best_state, best_aicc = state, aicc
        if best_state is None:  # every candidate failed to converge or fit
            best_state = _fit_sarima_css(y, (0, d, 0), (0, D, 0), m, True)
        return best_state

    def _fit_series(self, y: np.ndarray) -> dict:
        D = self._select_D(y)
        seasonally_differenced = y
        for _ in range(D):
            seasonally_differenced = (
                seasonally_differenced[self.m :] - seasonally_differenced[: -self.m]
            )
        d = self._select_d(seasonally_differenced)
        return self._search_orders(y, d, D)

    def _predict_series(self, state: dict, h: int) -> np.ndarray:
        return _forecast_sarima(state, h)

    def _predict_sigma(self, state: dict, h: int) -> np.ndarray:
        return _sarima_sigma(state, h)


# -- seasonal strength --------------------------------------------------------


def _centered_ma(y: np.ndarray, m: int) -> np.ndarray:
    """Centered moving average of span ``m`` (2xm weights when ``m`` is even).

    Returns an array the length of ``y`` with NaN where the window does not fit.
    """
    if m % 2:
        weights = np.ones(m) / m
    else:
        weights = np.concatenate([[0.5], np.ones(m - 1), [0.5]]) / m
    valid = np.convolve(y, weights, mode="valid")
    k = (len(weights) - 1) // 2
    trend = np.full(len(y), np.nan)
    trend[k : k + len(valid)] = valid
    return trend


def _mad_scale(x: np.ndarray) -> float:
    """Median-absolute-deviation scale estimate, normalized to match std on normals."""
    if len(x) == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _seasonal_strength(y: np.ndarray, m: int) -> float:
    """Robust seasonal strength ``max(0, 1 - s(remainder)² / s(detrended)²)`` in [0, 1].

    Uses a classical decomposition (centered moving-average trend, then
    cycle-subseries *medians*) rather than STL — the usual ``nsdiffs``
    statistic, computed with numpy only. The scale ``s`` is a normalized MAD
    rather than the standard deviation, so a handful of contaminated points
    cannot dominate the ratio. Returns 0 for series with no detrended
    variation at all (e.g. an exact linear trend).

    The seasonal component is deliberately *not* re-centered to mean zero:
    only scales enter the statistic and they are shift invariant, so
    centering would change nothing.
    """
    y = np.asarray(y, dtype=float)
    if m < 2 or len(y) < 2 * m:
        return 0.0
    trend = _centered_ma(y, m)
    ok = ~np.isnan(trend)
    detrended = y[ok] - trend[ok]
    idx = np.arange(len(y))[ok] % m
    # Cycle-subseries MEDIANS and a MAD-based scale, not means and variances:
    # both of the latter have a 0% breakdown point, so a single outlier lands
    # almost entirely in the remainder and drives Fs toward 0. One 6-sigma spike
    # was measured flipping Fs from 0.963 to 0.600 — enough to veto a seasonal
    # difference on a strongly seasonal series and triple the forecast error.
    centers = np.array(
        [np.median(detrended[idx == k]) if np.any(idx == k) else 0.0 for k in range(m)]
    )
    remainder = detrended - centers[idx]
    denom = _mad_scale(detrended) ** 2
    if denom <= 1e-12:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - _mad_scale(remainder) ** 2 / denom)))
