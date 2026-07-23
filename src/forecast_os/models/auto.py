"""AutoSelect: per-series model selection via walk-forward validation.

Candidate models (instances or registry-name strings, resolved with
:func:`get_model` lazily inside :meth:`fit`) are cross-validated on the
panel and scored per series; each winning model is refitted on the full
panel and :meth:`predict` stitches the winners' forecasts together. When
the panel is too short for the CV span, scoring falls back to a single
75/25 holdout per series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.base import BaseForecaster, _check_level
from ..core.registry import get_model, register
from ..core.types import ID_COL, TARGET_COL, TIME_COL, validate_panel
from ..evaluation.backtest import cross_validation
from ..evaluation.metrics import evaluate

__all__ = ["AutoSelect"]

_DEFAULT_CANDIDATES = ("naive", "drift", "ses", "holt", "theta", "auto_ets", "window_average")


@register("auto_select", family="ensemble")
class AutoSelect(BaseForecaster):
    """Pick the best candidate model per series by validation score."""

    def __init__(
        self,
        candidates=_DEFAULT_CANDIDATES,
        metric: str = "smape",
        val_h: int = 12,
        n_windows: int = 2,
    ):
        candidates = tuple(candidates)
        if not candidates:
            raise ValueError("AutoSelect requires at least one candidate model")
        if val_h < 1 or n_windows < 1:
            raise ValueError("val_h and n_windows must be positive integers")
        self.candidates = candidates
        self.metric = metric
        self.val_h = val_h
        self.n_windows = n_windows

    def clone(self) -> AutoSelect:
        """Deep-clone candidate instances; registry-name strings pass through."""
        params = self.get_params()
        params["candidates"] = tuple(
            c if isinstance(c, str) else c.clone() for c in self.candidates
        )
        return type(self)(**params)

    def fit(self, df: pd.DataFrame) -> AutoSelect:
        df = validate_panel(df)
        resolved = [get_model(c) if isinstance(c, str) else c.clone() for c in self.candidates]
        names = [m.name for m in resolved]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate candidate model names: {names}")

        sizes = df.groupby(ID_COL)[TARGET_COL].size()
        span = self.val_h * self.n_windows  # h + (n_windows - 1) * step_size, step = h
        if int(sizes.min()) > span:
            cv = cross_validation(df, resolved, h=self.val_h, n_windows=self.n_windows)
            scores = evaluate(cv, metrics=[self.metric], train_df=df)
        else:
            scores = self._holdout_scores(df, resolved)

        score_cols = [c for c in scores.columns if c not in (ID_COL, "metric")]
        winners: dict = {}
        for uid, row in scores.set_index(ID_COL)[score_cols].iterrows():
            vals = row.astype(float)
            winners[uid] = str(vals.idxmin()) if vals.notna().any() else names[0]

        by_name = {m.name: m for m in resolved}
        self._fitted_ = {
            name: by_name[name].clone().fit(df) for name in sorted(set(winners.values()))
        }
        self.best_models_ = winners
        self._is_fitted = True
        return self

    def _holdout_scores(self, df: pd.DataFrame, resolved: list) -> pd.DataFrame:
        """Score candidates on a per-series 75/25 holdout (short-panel fallback)."""
        n = df.groupby(ID_COL)[TARGET_COL].transform("size").to_numpy()
        hold = np.maximum(1, n // 4)
        pos = df.groupby(ID_COL).cumcount().to_numpy()
        train = df[pos < n - hold]
        test = df.loc[pos >= n - hold, [ID_COL, TIME_COL, TARGET_COL]].copy()
        test["_step"] = test.groupby(ID_COL).cumcount()
        h_max = int(hold.max())
        for m in resolved:
            pred = m.clone().fit(train).predict(h_max)
            pred["_step"] = pred.groupby(ID_COL).cumcount()
            pred = pred.rename(columns={"yhat": m.name})
            test = test.merge(pred[[ID_COL, "_step", m.name]], on=[ID_COL, "_step"], how="left")
        return evaluate(test.drop(columns="_step"), metrics=[self.metric], train_df=train)

    def predict(self, h: int, level: list[int] | None = None) -> pd.DataFrame:
        self._check_is_fitted()
        if not isinstance(h, (int, np.integer)) or h < 1:
            raise ValueError(f"h must be a positive integer, got {h!r}")
        _check_level(level)
        preds = {name: m.predict(h, level=level) for name, m in self._fitted_.items()}
        frames = [
            preds[name].loc[preds[name][ID_COL] == uid]
            for uid, name in self.best_models_.items()
        ]
        return pd.concat(frames, ignore_index=True)
