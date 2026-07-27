"""MSTL: additive decomposition with multiple seasonalities, plus a forecaster.

:func:`stl_decompose` splits a series into ``trend + Σ seasonal_p + remainder``
with one seasonal component per period (weekly *and* monthly, say). The
decomposition is **additive only** — there is no multiplicative mode; take a
log transform first if the seasonal amplitude scales with the level.

The algorithm is an STL-flavored iterative backfit implemented on plain numpy
(no statsmodels): the trend starts as a centered moving average of ``y`` and
then, for ``iterations`` sweeps over the periods in ascending order, each
seasonal component is re-estimated by cycle-subseries averaging (medians when
``robust=True``) of the series with the current trend and the *other* seasonal
components removed, centered to mean zero, and the trend is re-estimated as a
centered moving average of the deseasonalized series. Sweeping ascending
matters when periods are nested (7 and 28): the shorter component is removed
before the longer one is extracted, so the shared harmonic is not double
counted.

``remainder`` is defined as the residual ``y - (trend + Σ seasonal)``, so
recomposition returns the input series exactly rather than approximately: for
any series whose fitted part stays within a factor of two of the observations
the subtraction is exact in IEEE arithmetic (Sterbenz's lemma) and
``trend + Σ seasonal + remainder`` is bit-identical to ``y``. Only a series
straddling zero closely enough for ``trend + Σ seasonal`` to exceed ``2y`` can
push that reconstruction off by a fraction of an ulp.

:class:`MSTL` fits any registered per-series model to the deseasonalized
series ``trend + remainder`` and adds each seasonal component back at forecast
time, extended cyclically from its last full cycle.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ..core.base import PerSeriesForecaster
from ..core.exceptions import ForecastOSError
from ..core.registry import get_model, register

__all__ = ["STLResult", "stl_decompose", "MSTL"]


@dataclass
class STLResult:
    """Additive decomposition of one series.

    ``trend``, ``remainder`` and every array in ``seasonal`` have the same
    length as the input; ``periods`` lists the seasonal periods actually used
    (sorted, deduplicated, and with any period too long for the series
    dropped), and is exactly the key set of ``seasonal``.
    """

    trend: np.ndarray
    seasonal: dict[int, np.ndarray]
    remainder: np.ndarray
    periods: tuple[int, ...]

    def recompose(self) -> np.ndarray:
        """``trend + Σ seasonal + remainder``, which reproduces the input series.

        Bit-identical to the input except for the near-zero-crossing case
        described in the module docstring, where it is within an ulp.
        """
        return self.trend + _sum_seasonal(self.seasonal, len(self.trend)) + self.remainder


def _check_periods(periods: int | list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """Validate/normalize seasonal periods to a sorted, deduplicated tuple."""
    msg = f"periods must be an int or a sequence of ints >= 2, got {periods!r}"
    if isinstance(periods, (int, np.integer)) and not isinstance(periods, bool):
        raw: list = [periods]
    elif isinstance(periods, (list, tuple, set, frozenset, np.ndarray)):
        raw = list(periods)
    else:
        raise ForecastOSError(msg)
    out: list[int] = []
    for v in raw:
        try:
            iv = int(v)
        except (TypeError, ValueError) as exc:
            raise ForecastOSError(msg) from exc
        if iv != v or iv < 2:  # rejects 7.5, "7", True/False, and periods < 2
            raise ForecastOSError(msg)
        out.append(iv)
    if not out:
        raise ForecastOSError(msg)
    return tuple(sorted(set(out)))


def _check_iterations(iterations: int) -> int:
    try:
        it = int(iterations)
    except (TypeError, ValueError) as exc:
        raise ForecastOSError(f"iterations must be a positive int, got {iterations!r}") from exc
    if it < 1:
        raise ForecastOSError(f"iterations must be a positive int, got {iterations!r}")
    return it


def _usable_periods(periods: tuple[int, ...], n: int) -> tuple[int, ...]:
    """Drop periods without at least two full cycles of data, with a warning.

    Two cycles is the minimum for a cycle-subseries average to mean anything: at
    one cycle every cell holds a single observation, so the "seasonal" component
    memorizes the noise one point per cell, the remainder collapses toward zero,
    and the prediction intervals collapse with it (a nominal 95% interval was
    measured covering 9%). Requiring ``2 * p <= n`` makes every surviving period
    estimable from at least two observations per cell.
    """
    kept = tuple(p for p in periods if 2 * p <= n)
    dropped = [p for p in periods if 2 * p > n]
    if dropped:
        warnings.warn(
            f"ignoring seasonal period(s) {dropped}: a series of length {n} is "
            f"too short to estimate them (need 2 * period <= len(y))",
            UserWarning,
            stacklevel=3,
        )
    return kept


def _sum_seasonal(seasonal: dict[int, np.ndarray], n: int) -> np.ndarray:
    """Sum the seasonal components in ascending period order (fixed order = fixed rounding)."""
    total = np.zeros(n)
    for p in sorted(seasonal):
        total = total + seasonal[p]
    return total


def _centered_ma(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average of ``x``, same length as ``x``.

    An odd ``window`` uses flat weights; an even ``window`` uses the standard
    2xW average (half weight on the two end taps) so the window stays centered
    on an observation. The series is padded by edge replication, so the trend
    flattens toward the ends instead of returning NaN — a full-length trend is
    what makes the recomposition exact.
    """
    w = int(window)
    if w < 2:
        return np.asarray(x, dtype=float).copy()
    if w % 2 == 0:
        weights = np.ones(w + 1)
        weights[0] = weights[-1] = 0.5
        weights /= w
    else:
        weights = np.ones(w) / w
    half = len(weights) // 2
    padded = np.pad(np.asarray(x, dtype=float), half, mode="edge")
    return np.convolve(padded, weights, mode="valid")


def _cycle_seasonal(x: np.ndarray, period: int, robust: bool) -> np.ndarray:
    """Cycle-subseries average of ``x`` tiled to full length and centered to mean zero.

    Cell ``i`` is the mean (median when ``robust``) of ``x[i::period]``. The
    tiled pattern is centered by subtracting its own full-length mean, so the
    component has mean zero even when ``len(x)`` is not a multiple of
    ``period``; the constant it gives up is absorbed by the trend.
    """
    n = len(x)
    stat = np.median if robust else np.mean
    cells = np.array([float(stat(x[i::period])) for i in range(period)])
    s = cells[np.arange(n) % period]
    return s - float(np.mean(s))


def stl_decompose(
    y: np.ndarray,
    periods: int | list[int] | tuple[int, ...],
    iterations: int = 2,
    robust: bool = False,
) -> STLResult:
    """Additively decompose ``y`` into a trend, one seasonal per period, and a remainder.

    ``periods`` is a single int or a sequence of ints, each >= 2; duplicates are
    dropped and the periods are processed in ascending order. Periods that are
    not shorter than ``y`` are dropped with a ``UserWarning`` (with none left,
    the trend is ``y`` itself and the remainder is zero). ``robust=True``
    swaps the cycle-subseries mean for a median, which shrugs off isolated
    outliers. ``trend + Σ seasonal + remainder`` reproduces ``y`` exactly (see
    the module docstring for the one sub-ulp caveat).
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ForecastOSError(f"y must be a 1-d array, got shape {y.shape}")
    n = len(y)
    iterations = _check_iterations(iterations)
    kept = _usable_periods(_check_periods(periods), n)
    if not kept:
        return STLResult(trend=y.copy(), seasonal={}, remainder=np.zeros(n), periods=())

    window = max(kept)
    seasonal: dict[int, np.ndarray] = {p: np.zeros(n) for p in kept}
    trend = _centered_ma(y, window)
    for _ in range(iterations):
        for p in kept:
            others = _sum_seasonal({q: s for q, s in seasonal.items() if q != p}, n)
            seasonal[p] = _cycle_seasonal(y - trend - others, p, robust)
            trend = _centered_ma(y - _sum_seasonal(seasonal, n), window)
    total = _sum_seasonal(seasonal, n)
    return STLResult(
        trend=trend,
        seasonal=seasonal,
        remainder=y - (trend + total),
        periods=kept,
    )


@register("mstl", family="statistical")
class MSTL(PerSeriesForecaster):
    """Multi-seasonal decomposition + any base model on the deseasonalized series.

    ``fit`` decomposes each series with :func:`stl_decompose` (additive only)
    and fits ``base_model`` — any registered per-series forecaster, looked up
    by name with ``base_params`` as its constructor arguments — to
    ``trend + remainder``. ``predict`` adds each seasonal component back,
    extended cyclically from its last full cycle, so the base model never sees
    (or has to model) the seasonality.

    Prediction intervals come from the base model's own ``_predict_sigma``
    driven by MSTL's in-sample residual sigma: the seasonal components are
    treated as deterministic, so intervals reflect the deseasonalized model's
    uncertainty only.
    """

    def __init__(
        self,
        periods: int | list[int] | tuple[int, ...] = 7,
        base_model: str = "auto_ets",
        base_params: dict | None = None,
        iterations: int = 2,
    ):
        self.periods = _check_periods(periods)
        self.base_model = base_model
        self.base_params = base_params
        self.iterations = _check_iterations(iterations)
        # Gate on the *shortest* period, not the longest: periods too long for the
        # series are dropped with a warning by ``stl_decompose``. Gating on the
        # longest would make that warn-and-drop path unreachable through ``fit``
        # (e.g. periods=(7, 365) on 200 points would raise instead of forecasting
        # weekly), so the fit only requires enough data for the shortest period.
        # The base model is fitted on the deseasonalized series, so its own
        # requirement applies too — without this, a base needing more data than
        # MSTL silently returned all-NaN forecasts instead of erroring.
        base_min = 0
        try:
            base_min = int(self._make_base().min_train_size)
        except ForecastOSError:
            pass  # an unusable base_model reports itself at fit() time
        self.min_train_size = max(2 * min(self.periods), base_min)

    def _make_base(self) -> PerSeriesForecaster:
        """Instantiate the base forecaster, rejecting names the registry cannot serve."""
        try:
            base = get_model(self.base_model, **(self.base_params or {}))
        except (ValueError, TypeError) as exc:
            raise ForecastOSError(
                f"{self.name}: cannot build base_model {self.base_model!r} with "
                f"base_params {self.base_params!r}: {exc}"
            ) from exc
        if not isinstance(base, PerSeriesForecaster):
            raise ForecastOSError(
                f"{self.name}: base_model {self.base_model!r} must be a per-series "
                f"forecaster (a PerSeriesForecaster subclass)"
            )
        return base

    def _fit_series(self, y: np.ndarray) -> dict:
        res = stl_decompose(y, self.periods, iterations=self.iterations)
        total = _sum_seasonal(res.seasonal, len(y))
        deseason = y - total
        base = self._make_base()
        if len(deseason) < base.min_train_size:
            raise ForecastOSError(
                f"{self.name} with base_model={self.base_model!r} requires at least "
                f"{base.min_train_size} observations per series, got {len(deseason)}; "
                f"supply more history or choose a base_model with a lighter requirement"
            )
        try:
            base_state = base._fit_series(deseason)
        except ForecastOSError:
            raise
        except Exception as exc:  # a base model failing on its own terms
            raise ForecastOSError(
                f"{self.name}: base_model {self.base_model!r} failed to fit the "
                f"deseasonalized series: {exc}"
            ) from exc
        base_fitted = base_state.get("fitted")
        if base_fitted is None:  # base model without in-sample fits: one-step naive
            base_fitted = np.concatenate([[np.nan], deseason[:-1]])
        return {
            "fitted": np.asarray(base_fitted, dtype=float) + total,
            "trend_": res.trend,
            "seasonal_": res.seasonal,
            "remainder_": res.remainder,
            "periods_": res.periods,
            # last full cycle of each component: future step j reuses index j % p
            "cycles_": {p: res.seasonal[p][len(y) - p :].copy() for p in res.periods},
            "deseason_": deseason,
            "base_": base,
            "base_state_": base_state,
            "base_name": type(base).__name__,
        }

    def _predict_series(self, state: dict, h: int) -> np.ndarray:
        out = np.asarray(state["base_"]._predict_series(state["base_state_"], h), dtype=float)
        j = np.arange(h)
        for p, cycle in state["cycles_"].items():
            out = out + cycle[j % p]
        return out

    def _predict_sigma(self, state: dict, h: int) -> np.ndarray:
        base_state = {**state["base_state_"], "_sigma": state["_sigma"]}
        return np.asarray(state["base_"]._predict_sigma(base_state, h), dtype=float)
