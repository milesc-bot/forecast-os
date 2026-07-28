"""AutoSelect: per-series model selection via walk-forward validation.

Candidate models (instances or registry-name strings, resolved with
:func:`get_model` lazily inside :meth:`fit`) are cross-validated on the
panel and scored per series; each winning model is refitted on the full
panel and :meth:`predict` stitches the winners' forecasts together. When
the panel is too short for walk-forward validation — it does not clear the
CV span, or it clears it by so little that the first fold's training slice
cannot fit the candidates and the 75/25 split would train on more rows —
scoring falls back to a single 75/25 holdout per series.
"""

from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
import pandas as pd

from ..core.base import BaseForecaster, _check_level
from ..core.registry import get_model, register
from ..core.types import ID_COL, TARGET_COL, TIME_COL, infer_step, validate_panel
from ..evaluation.backtest import cross_validation
from ..evaluation.metrics import evaluate
from .baselines import SeasonalNaive
from .ets import AutoETS
from .theta import Theta

__all__ = ["AutoSelect"]

#: Default candidate pool when no seasonality is resolved (m <= 1).
_DEFAULT_CANDIDATES = ("naive", "drift", "ses", "holt", "theta", "auto_ets", "window_average")
#: Non-seasonal members of the default pool when a season length m > 1 is
#: resolved; seasonal SeasonalNaive/Theta/AutoETS instances are added to it.
_SEASONAL_BASE_CANDIDATES = ("naive", "drift", "ses", "window_average")

#: Rows a walk-forward training slice must have beyond the largest candidate
#: ``min_train_size`` before cross-validation is preferred to the holdout. The
#: margin is slack against a candidate that declares a ``min_train_size`` it
#: cannot really be fitted at (:class:`AutoETS` used to declare 3 while its
#: AICc screen needs ``n - k - 1 > 0``, i.e. 4). It only biases the *choice*
#: between two scoring schemes — a CV slice inside the margin is kept anyway
#: unless the holdout is strictly wider (see :meth:`AutoSelect.fit`), so no
#: panel is rejected for being one row short of it.
_CV_TRAIN_MARGIN = 1

#: Signed metrics (see :mod:`forecast_os.evaluation.metrics`): their best value
#: is ZERO, not minus infinity, so :meth:`AutoSelect.fit` ranks them by
#: ABSOLUTE value. A plain ``idxmin`` on the raw score would select the most
#: severely under-forecasting candidate. This mirrors the ranking rule
#: :meth:`forecast_os.Engine.compare` already applies to the same three names.
#: A NEW signed metric added to metrics.py must be listed here; anything not
#: listed is ranked as-is, which is correct for every lower-is-better point
#: metric.
_SIGNED_METRICS = frozenset({"bias", "pct_bias", "tracking_signal"})

#: Metrics that need ``lo``/``hi`` columns AutoSelect's scoring pass never
#: produces (it does not thread ``level`` into cross-validation), so they
#: cannot be scored at all. Rejected in ``__init__`` rather than blowing up
#: deep inside ``fit`` with advice the caller cannot act on. (``coverage`` is
#: additionally not lower-is-better.)
_INTERVAL_METRICS = frozenset({"coverage", "winkler", "pinball", "wis", "crps"})

#: Point metrics named in the error raised for the interval metrics.
_SUGGESTED_METRICS = "mae, rmse, mape, smape, mase, rmsse"

#: Season length implied by the root of an inferred pandas frequency string
#: (D->7, B->5, W->52, M/MS/ME->12, Q/QS/QE->4, H->24; anything else -> 1).
_SEASON_BY_FREQ = {
    "D": 7,
    "B": 5,
    "W": 52,
    "M": 12,
    "MS": 12,
    "ME": 12,
    "Q": 4,
    "QS": 4,
    "QE": 4,
    "H": 24,
    "h": 24,
}


def _infer_panel_season_length(df: pd.DataFrame) -> int:
    """Season length implied by the panel's modal per-series time step.

    Each series' step is inferred with :func:`~forecast_os.core.types.infer_step`
    and the most common one is mapped via D->7, B->5, W->52, M/MS/ME->12,
    Q/QS/QE->4, H->24. Any other step (irregular datetimes or numeric ``ds``)
    resolves to 1 (non-seasonal).
    """
    steps = [infer_step(g[TIME_COL]) for _, g in df.groupby(ID_COL, sort=True)]
    modal = Counter(steps).most_common(1)[0][0]
    if not isinstance(modal, str):
        return 1
    return _SEASON_BY_FREQ.get(modal.split("-")[0], 1)


def _scored_rows(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Per-series count of rows each candidate column is actually scored on.

    Mirrors :func:`~forecast_os.evaluation.metrics.evaluate`'s pairwise-complete
    rule (it drops rows where either ``y`` or the model column is NaN), so the
    counts say how much of the validation window each candidate covered.
    """
    scored = frame.dropna(subset=[TARGET_COL])
    return scored.groupby(ID_COL)[names].count()


@register("auto_select", family="ensemble")
class AutoSelect(BaseForecaster):
    """Pick the best candidate model per series by validation score.

    Parameters
    ----------
    candidates:
        Model instances or registry names to compete. ``None`` (default) uses
        a built-in pool that depends on the resolved season length ``m``: for
        ``m > 1`` it is naive, drift, ses, window_average plus
        ``SeasonalNaive(m)``, ``Theta(m)``, and ``AutoETS(m)``; for ``m <= 1``
        it is the non-seasonal pool (naive, drift, ses, holt, theta, auto_ets,
        window_average). An explicit candidate list is always used verbatim.
    metric:
        Scoring metric (default ``"mase"``: scale-free, and safe for
        zero-crossing returns-like series where smape saturates at 2).
        Must be a POINT metric. Lower-is-better metrics (``mae``, ``rmse``,
        ``mape``, ``smape``, ``mase``, ``rmsse``) are ranked as scored: the
        winner is the per-series argmin. The signed governance metrics
        (``bias``, ``pct_bias``, ``tracking_signal``) are ranked by ABSOLUTE
        value, since their best value is zero — the winner is the candidate
        closest to unbiased, not the argmin of the raw signed score (which is
        the most severely under-forecasting candidate). Note that
        ``tracking_signal`` and ``pct_bias`` score ``nan`` for a perfect
        forecast / an all-zero window; a candidate scoring ``nan`` is skipped,
        and a series where every candidate scores ``nan`` falls back to
        ranking on ``mae`` (as it does for any metric). The interval
        metrics (``coverage``, ``winkler``, ``pinball``, ``wis``) are
        rejected at construction: they cannot be scored because the
        validation pass never requests ``lo``/``hi`` columns.
        MASE/RMSSE scaling follows the M4 convention: it uses only the rows
        before the first validation cutoff (or before the holdout split),
        never the full panel, with seasonality equal to the resolved ``m``.
    val_h, n_windows:
        Walk-forward validation horizon and number of windows.
    season_length:
        Seasonal data needs a season length — without one no seasonal
        candidate enters the pool, so seasonal structure ends up in the
        residuals (mirroring :class:`~forecast_os.models.ets.AutoETS`'s
        guidance). ``"auto"`` (default): when ``candidates`` is None, infer
        ``m`` at fit time from the panel's modal inferred time step using the
        map D->7, B->5, W->52, M/MS/ME->12, Q/QS/QE->4, H->24 (anything else,
        including numeric ``ds``, -> 1); when candidates are given explicitly
        nothing is inferred (``m = 1``). An auto-inferred ``m`` is dropped back
        to 1 when the shortest training slice would not exceed it (e.g. 60
        weekly rows cannot support m=52). An explicit int is used as-is
        (``m <= 1`` disables seasonal candidates); ``None`` means 1 (disabled).
    """

    def __init__(
        self,
        candidates=None,
        metric: str = "mase",
        val_h: int = 12,
        n_windows: int = 2,
        season_length: int | str | None = "auto",
    ):
        if candidates is not None:
            candidates = tuple(candidates)
            if not candidates:
                raise ValueError("AutoSelect requires at least one candidate model")
        if metric in _INTERVAL_METRICS:
            raise ValueError(
                f"metric={metric!r} is an interval metric, but AutoSelect scores "
                f"candidates on point forecasts only — its validation pass never "
                f"requests lo/hi columns. Use a lower-is-better point metric "
                f"({_SUGGESTED_METRICS})."
            )
        if val_h < 1 or n_windows < 1:
            raise ValueError("val_h and n_windows must be positive integers")
        if season_length is not None and not isinstance(season_length, (int, np.integer)):
            if season_length != "auto":
                raise ValueError(
                    f"season_length must be an int, None, or 'auto', got {season_length!r}"
                )
        self.candidates = candidates
        self.metric = metric
        self.val_h = val_h
        self.n_windows = n_windows
        self.season_length = season_length

    def clone(self) -> AutoSelect:
        """Deep-clone candidate instances; registry-name strings pass through."""
        params = self.get_params()
        if self.candidates is not None:
            params["candidates"] = tuple(
                c if isinstance(c, str) else c.clone() for c in self.candidates
            )
        return type(self)(**params)

    def _resolve_season_length(self, df: pd.DataFrame) -> int:
        """Resolve ``season_length`` to a concrete ``m >= 1`` for this panel."""
        sl = self.season_length
        if sl is None:
            return 1
        if isinstance(sl, str):  # "auto" (validated in __init__)
            if self.candidates is not None:
                return 1  # inference only applies to the default candidate pool
            return _infer_panel_season_length(df)
        return max(int(sl), 1)

    def _candidate_specs(self, m: int) -> tuple:
        """Candidate models/names to compete given resolved season length ``m``."""
        if self.candidates is not None:
            return self.candidates
        if m <= 1:
            return _DEFAULT_CANDIDATES
        return _SEASONAL_BASE_CANDIDATES + (
            SeasonalNaive(season_length=m),
            Theta(season_length=m),
            AutoETS(season_length=m),
        )

    def _resolve_pool(self, df: pd.DataFrame, train_len: int) -> tuple[int, int, list]:
        """Season length, MASE seasonality and cloned candidates for a train slice.

        Seasonal candidates and seasonal MASE scaling both need strictly more
        than ``m`` training rows, so both depend on the shortest slice any
        candidate will be fitted on.
        """
        m = self._resolve_season_length(df)
        if m > 1 and train_len <= m and self.season_length == "auto":
            m = 1  # auto-inferred season doesn't fit the data; fall back
        scale_m = m if train_len > m else 1
        resolved = [
            get_model(c) if isinstance(c, str) else c.clone() for c in self._candidate_specs(m)
        ]
        return m, scale_m, resolved

    def fit(self, df: pd.DataFrame) -> AutoSelect:
        df = validate_panel(df)
        sizes = df.groupby(ID_COL)[TARGET_COL].size()
        span = self.val_h * self.n_windows  # h + (n_windows - 1) * step_size, step = h
        # Shortest training slice each scoring scheme would hand a candidate:
        # the first CV fold trains on ``n - span`` rows, the holdout on the
        # first 75%.
        cv_train_len = int((sizes - span).min())
        holdout_train_len = int((sizes - np.maximum(1, sizes // 4)).min())
        use_cv = int(sizes.min()) > span
        train_len = cv_train_len if use_cv else holdout_train_len
        m, scale_m, resolved = self._resolve_pool(df, train_len)
        if use_cv:
            needed = max(int(mdl.min_train_size) for mdl in resolved) + _CV_TRAIN_MARGIN
            if cv_train_len < needed and holdout_train_len > cv_train_len:
                # ``n`` just above the span leaves a first fold too short to fit
                # the pool on (n = span + 1 gave Drift a single row). That is
                # "too short for CV" as much as n <= span is: fall back to the
                # documented 75/25 holdout. The holdout is not always wider (it
                # keeps ``n - n // 4`` rows against the fold's ``n - span``), so
                # that has to be checked first: swapping a
                # feasible CV fold for a *narrower* holdout would reject panels
                # that fit (13 rows, val_h=1, n_windows=1, SeasonalNaive(12):
                # the fold trains on 12 rows, the holdout on 10).
                use_cv = False
                train_len = holdout_train_len
                m, scale_m, resolved = self._resolve_pool(df, train_len)
        names = [model.name for model in resolved]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate candidate model names: {names}")

        # "mae" rides along as the tie-breaker for series where the primary
        # metric is undefined (e.g. mase on a perfectly periodic series whose
        # seasonal-naive scale is exactly 0).
        metrics_list = [self.metric] if self.metric == "mae" else [self.metric, "mae"]
        if use_cv:
            cv = cross_validation(df, resolved, h=self.val_h, n_windows=self.n_windows)
            n = df.groupby(ID_COL)[TARGET_COL].transform("size").to_numpy()
            pos = df.groupby(ID_COL).cumcount().to_numpy()
            # MASE/RMSSE scaling must only see the first fold's training slice
            # (rows up to the first cutoff), matching the M4 convention.
            train_df = df[pos < n - span]
            scores = evaluate(cv, metrics=metrics_list, train_df=train_df, seasonality=scale_m)
            covered = _scored_rows(cv, names)
        else:
            scores, covered = self._holdout_scores(df, resolved, scale_m, metrics_list)

        score_cols = [c for c in scores.columns if c not in (ID_COL, "metric")]
        primary = scores[scores["metric"] == self.metric].set_index(ID_COL)[score_cols]
        fallback = scores[scores["metric"] == "mae"].set_index(ID_COL)[score_cols]
        winners: dict = {}
        short_covered: set = set()
        signed = self.metric in _SIGNED_METRICS
        for uid, row in primary.iterrows():
            vals = row.astype(float)
            ranked_signed = signed
            if not vals.notna().any() and uid in fallback.index:
                vals = fallback.loc[uid].astype(float)
                ranked_signed = False  # the mae fallback is already unsigned
            # ``evaluate`` drops NaN rows per model column, so a candidate that
            # forecast only part of its horizon is scored on the subset it did
            # produce -- typically the easy near-horizon steps -- and can beat a
            # candidate scored on the whole window. Those scores are not
            # comparable, so drop the short-covering candidates from the argmin
            # rather than let them compete. Candidates covering NOTHING already
            # score NaN and are excluded by ``idxmin``; they are left alone.
            cov = covered.loc[uid].reindex(vals.index) if uid in covered.index else None
            if cov is not None and cov.notna().any():
                short = cov.index[(cov > 0) & (cov < cov.max())]
                if len(short):
                    short_covered.update(map(str, short))
                    vals = vals.where(~vals.index.isin(short))
            # Signed metrics are best at zero, so rank them by magnitude: a raw
            # ``idxmin`` on ``bias`` would crown the biggest under-forecaster.
            ranking = vals.abs() if ranked_signed else vals
            winners[uid] = str(ranking.idxmin()) if ranking.notna().any() else names[0]
        if short_covered:
            warnings.warn(
                f"candidate model(s) {sorted(short_covered)} produced forecasts for "
                f"only part of the validation window (NaN for some horizon steps) "
                f"and were excluded from selection on the affected series; a "
                f"forecaster must return a value for every step of its horizon",
                UserWarning,
                stacklevel=2,
            )

        by_name = {m.name: m for m in resolved}
        self._fitted_ = {
            name: by_name[name].clone().fit(df) for name in sorted(set(winners.values()))
        }
        self.best_models_ = winners
        self._is_fitted = True
        return self

    def _holdout_scores(
        self,
        df: pd.DataFrame,
        resolved: list,
        seasonality: int,
        metrics_list: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Score candidates on a per-series 75/25 holdout (short-panel fallback).

        Returns ``(scores, covered)`` where ``covered`` counts the holdout rows
        each candidate was actually scored on (see :func:`_scored_rows`).
        """
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
        test = test.drop(columns="_step")
        scores = evaluate(
            test,
            metrics=metrics_list,
            train_df=train,
            seasonality=seasonality,
        )
        return scores, _scored_rows(test, [m.name for m in resolved])

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
