"""Ensemble meta-forecaster: combine member models' forecasts.

Members may be forecaster instances or registry-name strings; strings are
resolved with :func:`get_model` lazily inside :meth:`fit` (never at import
or construct time) so module registration order does not matter. Point
forecasts are combined by mean, median, or normalized weighted mean.
Interval bounds are combined the same way for ``mean``/``median`` and for
non-negative weights; under signed weights they are rebuilt from the
members' half-widths (see :meth:`Ensemble.predict`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.base import BaseForecaster, _check_level
from ..core.registry import get_model, register
from ..core.types import ID_COL, TIME_COL, validate_panel

__all__ = ["Ensemble"]

_MODES = ("mean", "median")


@register("ensemble", family="ensemble")
class Ensemble(BaseForecaster):
    """Combine member forecasts by mean, median, or weighted mean.

    ``weights`` are normalized by their sum and may be negative — signed
    combination weights are what Granger-Ramanathan / OLS forecast
    combination produces — as long as they are finite and sum to a positive
    number. Only the sum is constrained because a negative weight is a
    deliberate choice about the *point* forecast; the interval bounds are
    combined so that ``lo <= yhat <= hi`` holds regardless (see
    :meth:`predict`).
    """

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
        """Combined forecast; interval bounds stay ordered under signed weights.

        Applying a signed weight vector column-wise to ``lo``/``hi`` gives a
        combined width of ``sum_i w_i * (hi_i - lo_i)``, which turns negative
        as soon as a negatively-weighted member is wider than the positively
        weighted ones — yielding ``lo > yhat > hi``. So for the weighted mode
        the bands are rebuilt from the members' half-widths combined with
        ``|w_i|``: ``lo = yhat - sum_i |w_i| * (yhat_i - lo_i)``. That is
        arithmetically identical to the column-wise weighted mean whenever
        every weight is non-negative, and keeps ``lo <= yhat <= hi`` (and
        nesting across levels) by construction otherwise.
        """
        self._check_is_fitted()
        if not isinstance(h, (int, np.integer)) or h < 1:
            raise ValueError(f"h must be a positive integer, got {h!r}")
        levels = _check_level(level)
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
        if self._weights_ is not None and levels:
            abs_w = np.abs(self._weights_)
            yhat = out["yhat"].to_numpy(dtype=float)
            member_yhat = np.column_stack([p["yhat"].to_numpy(dtype=float) for p in preds])
            for lvl in levels:
                lo = np.column_stack([p[f"lo-{lvl}"].to_numpy(dtype=float) for p in preds])
                hi = np.column_stack([p[f"hi-{lvl}"].to_numpy(dtype=float) for p in preds])
                out[f"lo-{lvl}"] = yhat - (member_yhat - lo) @ abs_w
                out[f"hi-{lvl}"] = yhat + (hi - member_yhat) @ abs_w
        return out
