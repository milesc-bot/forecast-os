"""Base forecaster interfaces.

``BaseForecaster`` defines the contract every model in the registry satisfies:

- ``fit(df)`` takes a validated (unique_id, ds, y) panel and returns ``self``
- ``predict(h, level=None)`` returns a panel with columns
  ``unique_id, ds, yhat`` plus ``lo-{l}, hi-{l}`` per confidence level
- ``fitted_values()`` returns in-sample one-step fits for residual analysis
- ``get_params()/clone()`` support cross-validation refitting; constructor
  arguments MUST be stored as attributes of the same name.

``PerSeriesForecaster`` is the workhorse base for univariate models: it runs
the per-series groupby loop, generates future timestamps, and provides
Gaussian prediction intervals from in-sample residuals unless the subclass
supplies a model-specific ``_predict_sigma``.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .exceptions import ForecastOSError, NotFittedError
from .types import ID_COL, TARGET_COL, TIME_COL, future_ds, infer_step, validate_panel


def _check_level(level: list[int] | tuple[int, ...] | None) -> list[int]:
    if level is None:
        return []
    out = []
    for lvl in level:
        if not (0 < lvl < 100):
            raise ValueError(f"confidence levels must be in (0, 100), got {lvl}")
        out.append(int(lvl))
    return sorted(out)


class BaseForecaster(ABC):
    """Abstract base class for all forecast-os models."""

    #: Optional display name; used as the forecast column in cross-validation.
    alias: str | None = None

    @property
    def name(self) -> str:
        return self.alias or type(self).__name__

    def get_params(self) -> dict[str, Any]:
        """Constructor parameters, read back from same-named attributes."""
        sig = inspect.signature(type(self).__init__)
        params = {}
        for pname, p in sig.parameters.items():
            if pname == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if not hasattr(self, pname):
                raise ForecastOSError(
                    f"{type(self).__name__} must store constructor argument "
                    f"{pname!r} as attribute self.{pname} (required for clone())"
                )
            params[pname] = getattr(self, pname)
        return params

    def clone(self) -> "BaseForecaster":
        """A new unfitted instance with the same constructor parameters."""
        return type(self)(**self.get_params())

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseForecaster":
        """Fit the model on a (unique_id, ds, y) panel."""

    @abstractmethod
    def predict(self, h: int, level: list[int] | None = None) -> pd.DataFrame:
        """Forecast ``h`` steps ahead for every fitted series."""

    def fitted_values(self) -> pd.DataFrame:
        """In-sample fitted values as (unique_id, ds, y, fitted)."""
        raise NotImplementedError(f"{type(self).__name__} does not expose fitted values")

    def _check_is_fitted(self) -> None:
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError(f"{self.name} is not fitted; call fit() first")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        try:
            args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        except ForecastOSError:
            args = "..."
        return f"{type(self).__name__}({args})"


class PerSeriesForecaster(BaseForecaster):
    """Base class for univariate models fitted independently per series.

    Subclasses implement:

    - ``_fit_series(y: np.ndarray) -> dict`` returning the per-series state.
      If the state includes a ``"fitted"`` array (same length as ``y``, NaN
      for warm-up steps), it is used for residual-based intervals and
      :meth:`fitted_values`; otherwise a one-step-naive fallback is used.
    - ``_predict_series(state: dict, h: int) -> np.ndarray`` of length ``h``.
    - optionally ``_predict_sigma(state, h) -> np.ndarray`` for model-specific
      forecast standard errors (defaults to constant in-sample residual std).
    """

    #: Minimum observations per series required by the model.
    min_train_size: int = 1

    def fit(self, df: pd.DataFrame) -> "PerSeriesForecaster":
        df = validate_panel(df)
        self._series_state: dict[Any, dict] = {}
        for uid, g in df.groupby(ID_COL, sort=True):
            y = g[TARGET_COL].to_numpy(dtype=float)
            if len(y) < self.min_train_size:
                raise ForecastOSError(
                    f"{self.name} requires at least {self.min_train_size} "
                    f"observations per series; series {uid!r} has {len(y)}"
                )
            state = self._fit_series(y)
            if not isinstance(state, dict):
                raise ForecastOSError(
                    f"{type(self).__name__}._fit_series must return a dict state"
                )
            fitted = state.get("fitted")
            if fitted is None:
                fitted = np.concatenate([[np.nan], y[:-1]])
            fitted = np.asarray(fitted, dtype=float)
            resid = (y - fitted)[~np.isnan(y - fitted)]
            if resid.size >= 2:
                sigma = float(np.std(resid, ddof=1))
            elif len(y) >= 3:
                sigma = float(np.std(np.diff(y), ddof=1))
            else:
                sigma = 1.0
            state["fitted"] = fitted
            state["_y"] = y
            state["_ds"] = g[TIME_COL].to_numpy()
            state["_last_ds"] = g[TIME_COL].iloc[-1]
            state["_step"] = infer_step(g[TIME_COL])
            state["_sigma"] = max(sigma, 1e-12)
            self._series_state[uid] = state
        self._is_fitted = True
        return self

    def predict(self, h: int, level: list[int] | None = None) -> pd.DataFrame:
        self._check_is_fitted()
        if not isinstance(h, (int, np.integer)) or h < 1:
            raise ValueError(f"h must be a positive integer, got {h!r}")
        levels = _check_level(level)
        frames = []
        for uid, state in self._series_state.items():
            mean = np.asarray(self._predict_series(state, int(h)), dtype=float)
            if mean.shape != (h,):
                raise ForecastOSError(
                    f"{type(self).__name__}._predict_series returned shape "
                    f"{mean.shape}, expected ({h},)"
                )
            data: dict[str, Any] = {
                ID_COL: uid,
                TIME_COL: future_ds(state["_last_ds"], int(h), state["_step"]),
                "yhat": mean,
            }
            if levels:
                sigma = np.asarray(self._predict_sigma(state, int(h)), dtype=float)
                for lvl in levels:
                    z = float(stats.norm.ppf(0.5 + lvl / 200))
                    data[f"lo-{lvl}"] = mean - z * sigma
                    data[f"hi-{lvl}"] = mean + z * sigma
            frames.append(pd.DataFrame(data))
        return pd.concat(frames, ignore_index=True)

    def fitted_values(self) -> pd.DataFrame:
        self._check_is_fitted()
        frames = [
            pd.DataFrame(
                {
                    ID_COL: uid,
                    TIME_COL: state["_ds"],
                    TARGET_COL: state["_y"],
                    "fitted": state["fitted"],
                }
            )
            for uid, state in self._series_state.items()
        ]
        return pd.concat(frames, ignore_index=True)

    # -- hooks ---------------------------------------------------------------

    @abstractmethod
    def _fit_series(self, y: np.ndarray) -> dict:
        """Fit one series; return the state dict."""

    @abstractmethod
    def _predict_series(self, state: dict, h: int) -> np.ndarray:
        """Point forecast of length ``h`` for one series."""

    def _predict_sigma(self, state: dict, h: int) -> np.ndarray:
        """Forecast standard errors; default is constant residual std."""
        return np.full(h, state["_sigma"])
