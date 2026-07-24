"""The ForecastEngine facade: forecast, cross-validate, and compare models.

A thin, registry-aware entry point over the core contracts::

    engine = ForecastEngine(models=("auto_ets", "theta"), level=[80])
    engine.forecast(df, h=12)          # wide frame, one column set per model
    engine.cross_validate(df, h=12)    # walk-forward CV frame
    engine.compare(df, h=12)           # metric leaderboard (best model first)

Models are accepted as registry names or forecaster instances; strings are
resolved with :func:`~forecast_os.core.registry.get_model` at call time so the
engine never depends on import order.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .core.base import BaseForecaster
from .core.registry import get_model
from .core.types import ID_COL, TIME_COL, validate_panel
from .evaluation.backtest import cross_validation
from .evaluation.metrics import evaluate

__all__ = ["ForecastEngine"]


class ForecastEngine:
    """High-level facade over the model registry and evaluation harness.

    Parameters
    ----------
    models:
        Default models (registry names or forecaster instances) used when a
        method is called without an explicit ``models`` argument.
    level:
        Default confidence levels (e.g. ``[80, 95]``) for prediction
        intervals; ``None`` means point forecasts only.
    """

    def __init__(
        self,
        models: Sequence[BaseForecaster | str] = ("auto_ets",),
        level: list[int] | None = None,
    ):
        self.models = models
        self.level = level

    def _resolve(self, models: Sequence[BaseForecaster | str] | None) -> list[BaseForecaster]:
        models = self.models if models is None else models
        if isinstance(models, (str, BaseForecaster)):
            models = [models]
        resolved = [get_model(m) if isinstance(m, str) else m for m in models]
        if not resolved:
            raise ValueError("no models to run")
        names = [m.name for m in resolved]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate model names: {names}")
        return resolved

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        models: Sequence[BaseForecaster | str] | None = None,
        level: list[int] | None = None,
    ) -> pd.DataFrame:
        """Fit fresh clones of each model on ``df`` and forecast ``h`` steps.

        Returns a wide frame ``unique_id, ds, <model>[, <model>-lo-l,
        <model>-hi-l, ...]`` with one column set per model (the same renaming
        scheme cross-validation uses).
        """
        df = validate_panel(df)
        level = self.level if level is None else level
        frames = []
        for model in self._resolve(models):
            pred = model.clone().fit(df).predict(h, level=level)
            rename = {"yhat": model.name}
            for col in pred.columns:
                if col.startswith(("lo-", "hi-")):
                    rename[col] = f"{model.name}-{col}"
            pred = pred.rename(columns=rename)
            pred["_step"] = pred.groupby(ID_COL).cumcount()
            frames.append(pred)
        out = frames[0]
        for pred in frames[1:]:
            out = out.merge(pred.drop(columns=[TIME_COL]), on=[ID_COL, "_step"])
        return out.drop(columns="_step")

    def cross_validate(
        self,
        df: pd.DataFrame,
        h: int,
        n_windows: int = 3,
        step_size: int | None = None,
        models: Sequence[BaseForecaster | str] | None = None,
        level: list[int] | None = None,
    ) -> pd.DataFrame:
        """Walk-forward CV; thin wrapper over :func:`evaluation.backtest.cross_validation`."""
        level = self.level if level is None else level
        return cross_validation(
            df, self._resolve(models), h,
            n_windows=n_windows, step_size=step_size, level=level,
        )

    def compare(
        self,
        df: pd.DataFrame,
        h: int,
        n_windows: int = 3,
        metrics: Sequence[str] = ("mae", "rmse", "smape"),
        seasonality: int = 1,
        models: Sequence[BaseForecaster | str] | None = None,
        level: list[int] | None = None,
    ) -> pd.DataFrame:
        """Cross-validate and rank models on ``metrics`` (mean over series).

        Returns a leaderboard indexed by model name with one column per
        metric (best model first). Interval metrics (``coverage``,
        ``winkler``, ``pinball``, ``crps``) require ``level`` (falling back
        to the constructor default) and appear under their emitted per-level
        names, e.g. a ``coverage-80`` column for ``level=[80]``.

        Sorting: the board is ordered by the first metric's first emitted
        column. If that metric is ``coverage``, models sort by
        ``|value - level/100|`` ascending (closest to nominal coverage wins);
        otherwise plain ascending (lower is better).
        """
        df = validate_panel(df)
        metrics = list(metrics)
        level = self.level if level is None else level
        cv = cross_validation(df, self._resolve(models), h, n_windows=n_windows, level=level)
        scores = evaluate(cv, metrics=metrics, train_df=df, seasonality=seasonality)
        # map each requested metric to its emitted column(s), e.g.
        # "coverage" -> ["coverage-80", "coverage-95"]; point metrics map to themselves
        emitted = list(dict.fromkeys(scores["metric"]))
        columns: list[str] = []
        for name in metrics:
            per_level = [
                e for e in emitted if e.startswith(f"{name}-") and e[len(name) + 1 :].isdigit()
            ]
            columns += sorted(per_level, key=lambda e: int(e.rsplit("-", 1)[1])) or [name]
        board = scores.drop(columns=ID_COL).groupby("metric").mean().T
        board = board[columns]
        first = columns[0]
        if metrics[0] == "coverage":
            nominal = int(first.rsplit("-", 1)[1]) / 100.0
            board = board.reindex((board[first] - nominal).abs().sort_values().index)
        else:
            board = board.sort_values(first)
        board.index.name = "model"
        board.columns.name = None
        return board
