"""Ensemble meta-forecaster: combine member models' forecasts.

Members may be forecaster instances or registry-name strings; strings are
resolved with :func:`get_model` lazily inside :meth:`fit` (never at import
or construct time) so module registration order does not matter. Point
forecasts and interval bounds are combined the same way — mean, median, or
normalized weighted mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.base import BaseForecaster
from ..core.registry import get_model, register
from ..core.types import ID_COL, TIME_COL, validate_panel

__all__ = ["Ensemble"]

_MODES = ("mean", "median")


@register("ensemble", family="ensemble")
class Ensemble(BaseForecaster):
    """Combine member forecasts by mean, median, or weighted mean."""

    def __init__(self, models=("naive", "ses", "drift"), mode="mean", weights=None):
        models = tuple(models)
        if not models:
            raise ValueError("Ensemble requires at least one member model")
        if mode not in _MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {_MODES}")
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            if w.ndim != 1 or len(w) != len(models):
                raise ValueError(
                    f"weights must have one entry per model ({len(models)}), got {w.shape}"
                )
            if not np.all(np.isfinite(w)) or w.sum() <= 0:
                raise ValueError("weights must be finite with a positive sum")
            if mode == "median":
                raise ValueError("weights are only supported with mode='mean'")
        self.models = models
        self.mode = mode
        self.weights = weights

    def clone(self) -> Ensemble:
        """Deep-clone member instances; registry-name strings pass through."""
        params = self.get_params()
        params["models"] = tuple(
            m if isinstance(m, str) else m.clone() for m in self.models
        )
        return type(self)(**params)

    def fit(self, df: pd.DataFrame) -> Ensemble:
        df = validate_panel(df)
        members = [get_model(m) if isinstance(m, str) else m.clone() for m in self.models]
        self._members_ = [m.fit(df) for m in members]
        if self.weights is not None:
            w = np.asarray(self.weights, dtype=float)
            self._weights_ = w / w.sum()
        else:
            self._weights_ = None
        self._is_fitted = True
        return self

    def predict(self, h: int, level: list[int] | None = None) -> pd.DataFrame:
        self._check_is_fitted()
        preds = [
            m.predict(h, level=level)
            .sort_values([ID_COL, TIME_COL], kind="stable")
            .reset_index(drop=True)
            for m in self._members_
        ]
        out = preds[0][[ID_COL, TIME_COL]].copy()
        value_cols = [c for c in preds[0].columns if c not in (ID_COL, TIME_COL)]
        for col in value_cols:
            stack = np.column_stack([p[col].to_numpy(dtype=float) for p in preds])
            if self._weights_ is not None:
                out[col] = stack @ self._weights_
            elif self.mode == "median":
                out[col] = np.median(stack, axis=1)
            else:
                out[col] = stack.mean(axis=1)
        return out
