"""The ForecastEngine facade: forecast, cross-validate, and compare models.

A thin, registry-aware entry point over the core contracts::

    engine = ForecastEngine(models=("auto_ets", "theta"), level=[80])
    engine.forecast(df, h=12)          # wide frame, one column set per model
    engine.cross_validate(df, h=12)    # walk-forward CV frame
    engine.compare(df, h=12)           # metric leaderboard (best model first)

Models are accepted as registry names, forecaster instances, or parameterized
``(name, params)`` specs such as ``("ridge_lag", {"lags": 12})``; names and
specs are resolved with :func:`~forecast_os.core.registry.get_model` at call
time so the engine never depends on import order.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import pandas as pd

from .core.base import BaseForecaster
from .core.exceptions import ForecastOSError
from .core.registry import get_model
from .core.types import ID_COL, TARGET_COL, TIME_COL, validate_panel
from .evaluation.backtest import cross_validation
from .evaluation.metrics import evaluate

__all__ = ["ForecastEngine"]

#: A model argument: registry name, instance, or ``(name, params)`` spec.
ModelLike = BaseForecaster | str | tuple[str, dict[str, Any]]


def _is_spec(m: Any) -> bool:
    """True for a parameterized ``(name, params)`` model spec."""
    return (
        isinstance(m, tuple)
        and len(m) == 2
        and isinstance(m[0], str)
        and isinstance(m[1], dict)
    )


class ForecastEngine:
    """High-level facade over the model registry and evaluation harness.

    Parameters
    ----------
    models:
        Default models used when a method is called without an explicit
        ``models`` argument. Each entry may be a registry name, a forecaster
        instance, or a parameterized ``(name, params)`` spec such as
        ``("ridge_lag", {"lags": 12})``.
    level:
        Default confidence levels (e.g. ``[80, 95]``) for prediction
        intervals; ``None`` means point forecasts only.
    """

    def __init__(
        self,
        models: Sequence[ModelLike] = ("auto_ets",),
        level: list[int] | None = None,
    ):
        self.models = models
        self.level = level

    def _resolve(self, models: Sequence[ModelLike] | None) -> list[BaseForecaster]:
        models = self.models if models is None else models
        if isinstance(models, (str, BaseForecaster)) or _is_spec(models):
            models = [models]
        resolved = []
        for m in models:
            if isinstance(m, str):
                resolved.append(get_model(m))
            elif _is_spec(m):
                name, params = m
                resolved.append(get_model(name, **params))
            elif isinstance(m, BaseForecaster):
                resolved.append(m)
            else:
                raise ValueError(
                    f"cannot resolve model spec {m!r}; expected a registry name, "
                    f"a BaseForecaster instance, or a (name, params) tuple"
                )
        if not resolved:
            raise ValueError("no models to run")
        names = [m.name for m in resolved]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate model names: {names}")
        return resolved

    @staticmethod
    def _display_name(model: BaseForecaster) -> str:
        """Leaderboard name: the registry name of the model's own class, if any.

        ``register()`` stamps ``registry_name`` on the decorated class, so the
        lookup uses the class ``__dict__`` (not ``getattr``) to avoid picking
        up an inherited name from an unregistered subclass. Falls back to
        ``model.name``.
        """
        return type(model).__dict__.get("registry_name") or model.name

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        models: Sequence[ModelLike] | None = None,
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
        models: Sequence[ModelLike] | None = None,
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
        models: Sequence[ModelLike] | None = None,
        level: list[int] | None = None,
    ) -> pd.DataFrame:
        """Cross-validate and rank models on ``metrics`` (mean over series).

        Returns a leaderboard with one column per metric (best model first).
        The board is indexed by each model's registry name when its class was
        registered (falling back to ``model.name``), so an index entry can be
        fed straight back into :meth:`forecast`. Interval metrics
        (``coverage``, ``winkler``, ``pinball``, ``crps``) require ``level``
        (falling back to the constructor default) and appear under their
        emitted per-level names, e.g. a ``coverage-80`` column for
        ``level=[80]``.

        Failure tolerance: each model is cross-validated independently; a
        model whose fit/predict raises is dropped with a ``UserWarning``
        (``"model X failed: ..."``) and the board is built from the
        survivors. If every model fails, :class:`ForecastOSError` is raised.

        Sorting: the board is ordered by the first metric's first emitted
        column. If that metric is ``coverage``, models sort by
        ``|value - level/100|`` ascending (closest to nominal coverage wins).
        If it is a signed governance metric (``bias``, ``pct_bias``,
        ``tracking_signal``), models sort by ``|value|`` ascending (closest
        to zero wins — a plain ascending sort would crown the most
        negatively biased model). Otherwise plain ascending (lower is
        better).
        """
        df = validate_panel(df)
        metrics = list(metrics)
        level = self.level if level is None else level
        resolved = self._resolve(models)
        board_names = {m.name: self._display_name(m) for m in resolved}
        cv = None
        failures: list[str] = []
        for model in resolved:
            name = self._display_name(model)
            try:
                cv_model = cross_validation(df, [model], h, n_windows=n_windows, level=level)
            except Exception as exc:
                warnings.warn(f"model {name} failed: {exc}", stacklevel=2)
                failures.append(f"{name}: {exc}")
                continue
            if cv is None:
                cv = cv_model
            else:
                cv = cv.merge(
                    cv_model.drop(columns=[TARGET_COL]), on=[ID_COL, "cutoff", TIME_COL]
                )
        if cv is None:
            raise ForecastOSError("all models failed in compare(): " + "; ".join(failures))
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
        elif metrics[0] in ("bias", "pct_bias", "tracking_signal"):
            # signed governance metrics: closest to zero wins, mirroring the
            # coverage rule -- ascending would rank a -31.1-bias model above
            # a -0.5-bias one
            board = board.reindex(board[first].abs().sort_values().index)
        else:
            board = board.sort_values(first)
        board.index = board.index.map(lambda n: board_names.get(n, n))
        board.index.name = "model"
        board.columns.name = None
        return board
