"""Exponential smoothing models: SES, Holt, Holt-Winters, and AutoETS.

Smoothing parameters left as ``None`` are optimized per series by minimizing
the one-step in-sample SSE (bounded to [0.01, 0.99]); optimized values live in
the per-series state (``alpha_`` etc.), never on ``self``, so a fitted model
holds independent parameters for every series in the panel.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from ..core.base import PerSeriesForecaster
from ..core.exceptions import ForecastOSError
from ..core.registry import register

__all__ = ["SES", "Holt", "HoltWinters", "AutoETS"]

_BOUNDS = (0.01, 0.99)


def _ses_path(y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Run the SES recursion; return (one-step fitted values, final level)."""
    fitted = np.full(len(y), np.nan)
    level = y[0]
    for t in range(1, len(y)):
        fitted[t] = level
        level = alpha * y[t] + (1 - alpha) * level
    return fitted, float(level)


def _fit_ses(y: np.ndarray, alpha: float | None = None) -> tuple[float, float, np.ndarray]:
    """Fit SES on one series; return (alpha, final level, fitted values)."""
    if alpha is None:
        def sse(a: float) -> float:
            fitted, _ = _ses_path(y, a)
            return float(np.nansum((y - fitted) ** 2))

        alpha = float(minimize_scalar(sse, bounds=_BOUNDS, method="bounded").x)
    fitted, level = _ses_path(y, float(alpha))
    return float(alpha), level, fitted


@register("ses", family="statistical")
class SES(PerSeriesForecaster):
    """Simple exponential smoothing with SSE-optimized alpha by default."""

    def __init__(self, alpha: float | None = None):
        if alpha is not None and not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha

    def _fit_series(self, y: np.ndarray) -> dict:
        alpha, level, fitted = _fit_ses(y, self.alpha)
        return {"fitted": fitted, "alpha_": alpha, "level_": level}

    def _predict_series(self, state: dict, h: int) -> np.ndarray:
        return np.full(h, state["level_"])

    def _predict_sigma(self, state: dict, h: int) -> np.ndarray:
        k = np.arange(1, h + 1)
        return state["_sigma"] * np.sqrt(1 + (k - 1) * state["alpha_"] ** 2)


@register("holt", family="statistical")
class Holt(PerSeriesForecaster):
    """Holt's linear trend method (double exponential smoothing)."""

    min_train_size = 3

    def __init__(self, alpha: float | None = None, beta: float | None = None):
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def _path(y: np.ndarray, alpha: float, beta: float) -> tuple[np.ndarray, float, float]:
        fitted = np.full(len(y), np.nan)
        level, trend = y[0], y[1] - y[0]
        for t in range(1, len(y)):
            fitted[t] = level + trend
            new_level = alpha * y[t] + (1 - alpha) * (level + trend)
            trend = beta * (new_level - level) + (1 - beta) * trend
            level = new_level
        return fitted, float(level), float(trend)

    def _fit_series(self, y: np.ndarray) -> dict:
        params = {"alpha": self.alpha, "beta": self.beta}
        free = [name for name, v in params.items() if v is None]
        if free:
            x0 = {"alpha": 0.3, "beta": 0.1}

            def sse(x: np.ndarray) -> float:
                p = {**params, **dict(zip(free, x))}
                fitted, _, _ = self._path(y, p["alpha"], p["beta"])
                return float(np.nansum((y - fitted) ** 2))

            res = minimize(
                sse, x0=[x0[n] for n in free], bounds=[_BOUNDS] * len(free), method="L-BFGS-B"
            )
            params.update({n: float(v) for n, v in zip(free, res.x)})
        fitted, level, trend = self._path(y, params["alpha"], params["beta"])
        return {
            "fitted": fitted,
            "alpha_": params["alpha"],
            "beta_": params["beta"],
            "level_": level,
            "trend_": trend,
        }

    def _predict_series(self, state: dict, h: int) -> np.ndarray:
        return state["level_"] + np.arange(1, h + 1) * state["trend_"]


@register("holt_winters", family="statistical")
class HoltWinters(PerSeriesForecaster):
    """Holt-Winters seasonal smoothing with additive or multiplicative season."""

    def __init__(
        self,
        season_length: int = 7,
        seasonal: str = "add",
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
    ):
        if seasonal not in ("add", "mul"):
            raise ValueError(f"seasonal must be 'add' or 'mul', got {seasonal!r}")
        if season_length < 2:
            raise ValueError(f"season_length must be >= 2, got {season_length}")
        self.season_length = season_length
        self.seasonal = seasonal
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.min_train_size = 2 * season_length

    def _path(
        self, y: np.ndarray, alpha: float, beta: float, gamma: float
    ) -> tuple[np.ndarray, float, float, np.ndarray]:
        m = self.season_length
        mul = self.seasonal == "mul"
        n = len(y)
        mean1 = float(np.mean(y[:m]))
        mean2 = float(np.mean(y[m : 2 * m]))
        level, trend = mean1, (mean2 - mean1) / m
        if mul:
            base = mean1 if abs(mean1) > 1e-12 else 1e-12
            seasons = list(y[:m] / base)
        else:
            seasons = list(y[:m] - mean1)
        fitted = np.full(n, np.nan)
        for t in range(m, n):
            s = seasons[t - m]
            if mul:
                s = s if abs(s) > 1e-12 else 1e-12
                fitted[t] = (level + trend) * s
                new_level = alpha * (y[t] / s) + (1 - alpha) * (level + trend)
                lvl = new_level if abs(new_level) > 1e-12 else 1e-12
                new_season = gamma * (y[t] / lvl) + (1 - gamma) * s
            else:
                fitted[t] = level + trend + s
                new_level = alpha * (y[t] - s) + (1 - alpha) * (level + trend)
                new_season = gamma * (y[t] - new_level) + (1 - gamma) * s
            trend = beta * (new_level - level) + (1 - beta) * trend
            level = new_level
            seasons.append(new_season)
        return fitted, float(level), float(trend), np.asarray(seasons[-m:], dtype=float)

    def _fit_series(self, y: np.ndarray) -> dict:
        if self.seasonal == "mul" and np.any(y <= 0):
            raise ForecastOSError(
                "HoltWinters(seasonal='mul') requires strictly positive values; "
                "use seasonal='add' or a LogTransform"
            )
        params = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma}
        free = [name for name, v in params.items() if v is None]
        if free:
            x0 = {"alpha": 0.3, "beta": 0.05, "gamma": 0.1}

            def sse(x: np.ndarray) -> float:
                p = {**params, **dict(zip(free, x))}
                fitted, *_ = self._path(y, p["alpha"], p["beta"], p["gamma"])
                return float(np.nansum((y - fitted) ** 2))

            res = minimize(
                sse, x0=[x0[n] for n in free], bounds=[_BOUNDS] * len(free), method="L-BFGS-B"
            )
            params.update({n: float(v) for n, v in zip(free, res.x)})
        fitted, level, trend, seasons = self._path(
            y, params["alpha"], params["beta"], params["gamma"]
        )
        return {
            "fitted": fitted,
            "alpha_": params["alpha"],
            "beta_": params["beta"],
            "gamma_": params["gamma"],
            "level_": level,
            "trend_": trend,
            "season_": seasons,
        }

    def _predict_series(self, state: dict, h: int) -> np.ndarray:
        k = np.arange(1, h + 1)
        base = state["level_"] + k * state["trend_"]
        s = state["season_"][(k - 1) % self.season_length]
        return base * s if self.seasonal == "mul" else base + s


@register("auto_ets", family="statistical")
class AutoETS(PerSeriesForecaster):
    """Automatic ETS selection (SES / Holt / Holt-Winters) by per-series AICc."""

    min_train_size = 3

    def __init__(self, season_length: int | None = None):
        self.season_length = season_length

    def _candidates(self, y: np.ndarray) -> list[tuple[PerSeriesForecaster, int]]:
        # k = number of free smoothing params + number of initial states
        cands: list[tuple[PerSeriesForecaster, int]] = [(SES(), 2), (Holt(), 4)]
        m = self.season_length
        if m and m > 1 and 2 * m <= len(y):
            cands.append((HoltWinters(season_length=m, seasonal="add"), 5 + m))
            if np.all(y > 0):
                cands.append((HoltWinters(season_length=m, seasonal="mul"), 5 + m))
        return cands

    def _fit_series(self, y: np.ndarray) -> dict:
        n = len(y)
        best = None
        for model, k in self._candidates(y):
            if n - k - 1 <= 0:
                continue
            try:
                st = model._fit_series(y)
            except ForecastOSError:
                continue
            resid = y - st["fitted"]
            resid = resid[~np.isnan(resid)]
            if resid.size == 0:
                continue
            sse = max(float(np.sum(resid**2)), 1e-300)
            aicc = n * np.log(sse / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)
            if best is None or aicc < best[0]:
                best = (aicc, model, st)
        if best is None:
            raise ForecastOSError(f"{self.name}: no ETS candidate could be fitted")
        aicc, model, st = best
        return {
            "fitted": st["fitted"],
            "winner_": model,
            "winner_state_": st,
            "winner_name": type(model).__name__,
            "aicc_": aicc,
        }

    def _predict_series(self, state: dict, h: int) -> np.ndarray:
        return state["winner_"]._predict_series(state["winner_state_"], h)

    def _predict_sigma(self, state: dict, h: int) -> np.ndarray:
        winner_state = {**state["winner_state_"], "_sigma": state["_sigma"]}
        return state["winner_"]._predict_sigma(winner_state, h)
