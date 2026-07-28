"""The compute layer behind the terminal screens: pure functions, plain frames.

Every screen in the terminal UI renders the output of one function here —
:func:`dashboard_rows` (fast 1-step outlook + sparklines), :func:`forecast_frame`
(history + fan-chart forecast), :func:`leaderboard_frame` (model comparison),
:func:`governance_frame` (per-cutoff accuracy/bias/coverage), and
:func:`alerts_fired` (alert rules evaluated against fresh forecasts). They take
a contract panel plus the workspace ``settings`` dict and return plain pandas
DataFrames (or strings), so all of it is unit-testable without textual or a
running app.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pandas as pd

from ..core.exceptions import ForecastOSError
from ..core.registry import get_model
from ..core.types import ID_COL, REQUIRED_COLS, TARGET_COL
from ..engine import ForecastEngine
from ..evaluation.backtest import cross_validation
from ..evaluation.metrics import evaluate

__all__ = [
    "DEFAULT_LEADERBOARD_MODELS",
    "ComputeFailure",
    "alerts_fired",
    "dashboard_rows",
    "forecast_frame",
    "governance_frame",
    "leaderboard_frame",
    "model_params",
    "sparkline",
]


class ComputeFailure:
    """Sentinel result for a worker whose compute raised.

    Screens store this in place of a DataFrame so a failed leaderboard or
    governance run renders "unavailable: <reason>" instead of being
    indistinguishable from "not computed yet" (which would re-request
    forever). Cleared back to ``None`` only by an explicit refresh.
    """

    __slots__ = ("reason",)

    def __init__(self, reason: str):
        self.reason = reason

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ComputeFailure({self.reason!r})"

#: Default leaderboard candidates: a seasonal baseline, two strong
#: statistical models, and one ML autoregression.
DEFAULT_LEADERBOARD_MODELS = ("seasonal_naive", "theta", "auto_ets", "ridge_lag")

_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"
_SPARK_NAN = "·"


def model_params(model: str, settings: dict) -> dict:
    """Constructor params for registry ``model`` implied by workspace ``settings``.

    Currently that is the workspace season alone, passed through only when it is
    set and the model's constructor accepts it — so ``naive`` stays
    parameterless while ``theta``/``auto_ets``/``seasonal_naive``/``ridge_lag``
    pick the season up.

    Models spell that parameter differently: ``sarima``/``auto_sarima`` take
    ``m`` and ``mstl`` takes ``periods``. Matching only on ``season_length``
    silently left those three on their class defaults (``m=12`` on daily data),
    so the workspace setting is mapped onto whichever name the model uses.
    """
    season_length = settings.get("season_length")
    if season_length is None:
        return {}
    probe = get_model(model)  # constructors only store attributes; cheap
    accepted = inspect.signature(type(probe).__init__).parameters
    for name in ("season_length", "m", "periods"):
        if name in accepted:
            return {name: int(season_length)}
    return {}


def sparkline(values, width: int = 12) -> str:
    """The last ``width`` values as a unicode block-character sparkline.

    Values are min-max scaled over the window's finite entries; NaN renders
    as ``·`` and a constant window renders as a flat mid-height line. A
    non-positive ``width`` asks for no values and renders as ``""`` (slicing
    with ``[-0:]`` would otherwise return the whole series).
    """
    if width <= 0:
        return ""
    vals = np.asarray(list(values), dtype=float)[-width:]
    if vals.size == 0:
        return ""
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return _SPARK_NAN * vals.size
    lo, hi = float(finite.min()), float(finite.max())
    span = hi - lo
    chars = []
    for v in vals:
        if not math.isfinite(v):
            chars.append(_SPARK_NAN)
        elif span < 1e-12:
            chars.append(_SPARK_BLOCKS[3])
        else:
            chars.append(_SPARK_BLOCKS[round((v - lo) / span * (len(_SPARK_BLOCKS) - 1))])
    return "".join(chars)


def dashboard_rows(panel: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """One dashboard row per series: last actual, 1-step outlook, delta, sparkline.

    The 1-step forecast uses ``theta`` (fast, trend-aware) with the settings
    ``season_length``; any series whose fit fails — too short, NaN targets —
    falls back to the naive forecast (``next = last``) instead of raising, so
    one bad series never takes the dashboard down. ``delta_pct`` is the
    percent change from ``last`` to ``next`` (NaN when ``last`` is ~0).
    Columns: ``unique_id, last, next, delta_pct, spark``.
    """
    season_length = settings.get("season_length")
    season_length = None if season_length is None else int(season_length)
    rows = []
    for uid, g in panel.groupby(ID_COL, sort=True):
        y = g[TARGET_COL].to_numpy(dtype=float)
        last = float(y[-1])
        try:
            model = get_model("theta", season_length=season_length)
            nxt = float(model.fit(g[list(REQUIRED_COLS)]).predict(1)["yhat"].iloc[0])
        except Exception:
            nxt = last  # per-series naive fallback; never raise for one bad series
        delta = (nxt - last) / abs(last) * 100.0 if abs(last) > 1e-12 else float("nan")
        rows.append(
            {
                ID_COL: uid,
                "last": last,
                "next": nxt,
                "delta_pct": delta,
                "spark": sparkline(y),
            }
        )
    return pd.DataFrame(rows, columns=[ID_COL, "last", "next", "delta_pct", "spark"])


def forecast_frame(
    panel: pd.DataFrame, series: str, settings: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """History and forecast frames for one series, per the workspace settings.

    Fits ``settings["model"]`` (with :func:`model_params`) on the series and
    predicts ``settings["h"]`` steps at ``settings["level"]``. Returns
    ``(history, forecast)``: history is the series' contract rows; forecast
    has columns ``unique_id, ds, yhat, lo, hi`` (interval bounds renamed from
    their ``lo-{level}``/``hi-{level}`` originals). An unknown ``series``
    raises :class:`ForecastOSError`.
    """
    history = panel.loc[panel[ID_COL] == series, list(REQUIRED_COLS)].reset_index(drop=True)
    if history.empty:
        known = ", ".join(sorted(panel[ID_COL].astype(str).unique())[:8])
        raise ForecastOSError(f"unknown series {series!r}; panel has: {known}")
    name = settings.get("model", "auto_ets")
    h = int(settings.get("h", 6))
    level = int(settings.get("level", 80))
    model = get_model(name, **model_params(name, settings))
    forecast = model.fit(history).predict(h, level=[level])
    forecast = forecast.rename(columns={f"lo-{level}": "lo", f"hi-{level}": "hi"})
    return history, forecast


def leaderboard_frame(
    panel: pd.DataFrame,
    settings: dict,
    models: list[str] | tuple[str, ...] | None = None,
    n_windows: int = 2,
) -> pd.DataFrame:
    """Model leaderboard on ``panel``: mase + interval coverage, best first.

    Cross-validates ``models`` (default :data:`DEFAULT_LEADERBOARD_MODELS`,
    each parameterized via :func:`model_params`) at the settings horizon and
    level, scoring ``mase`` and ``coverage``. Failure-tolerant by way of
    :meth:`~forecast_os.engine.ForecastEngine.compare`: a model that cannot
    fit is dropped with a warning rather than sinking the board. Returns a
    tidy frame with a ``model`` column plus one column per emitted metric
    (e.g. ``mase``, ``coverage-80``).
    """
    names = list(models) if models is not None else list(DEFAULT_LEADERBOARD_MODELS)
    h = int(settings.get("h", 6))
    # size ridge_lag's default lags to the smallest CV training window
    # (min_train_size is lags + 10) so short monthly panels keep the candidate
    first_train = int(panel.groupby(ID_COL).size().min()) - h * n_windows
    specs = []
    for name in names:
        params = model_params(name, settings)
        if name == "ridge_lag" and "lags" not in params:
            params["lags"] = max(1, min(12, first_train - 10))
        specs.append((name, params))
    engine = ForecastEngine(models=specs, level=[int(settings.get("level", 80))])
    board = engine.compare(
        panel,
        h=h,
        n_windows=n_windows,
        metrics=["mase", "coverage"],
        seasonality=int(settings.get("season_length") or 1),
    )
    return board.reset_index()


def governance_frame(
    panel: pd.DataFrame, settings: dict, n_windows: int = 2
) -> pd.DataFrame:
    """Per-(series, cutoff) governance table for the settings model.

    Walk-forward cross-validates ``settings["model"]`` and scores every
    ``(unique_id, cutoff)`` group on ``mase`` (accuracy), ``pct_bias``
    (signed sandbagging direction: negative = forecast ran low), and
    interval ``coverage`` at the settings level. Returns a tidy frame with
    columns ``unique_id, cutoff, mase, pct_bias, coverage``.
    """
    name = settings.get("model", "auto_ets")
    level = int(settings.get("level", 80))
    model = get_model(name, **model_params(name, settings))
    cv = cross_validation(
        panel, [model], int(settings.get("h", 6)), n_windows=n_windows, level=[level]
    )
    tidy = evaluate(
        cv,
        metrics=["mase", "pct_bias", "coverage"],
        by="cutoff",
        train_df=panel,
        seasonality=int(settings.get("season_length") or 1),
    )
    value_col = [c for c in tidy.columns if c not in (ID_COL, "cutoff", "metric")][0]
    out = (
        tidy.pivot(index=[ID_COL, "cutoff"], columns="metric", values=value_col)
        .reset_index()
        .rename(columns={f"coverage-{level}": "coverage"})
    )
    out.columns.name = None
    return out[[ID_COL, "cutoff", "mase", "pct_bias", "coverage"]]


def alerts_fired(
    panel: pd.DataFrame,
    forecasts: pd.DataFrame,
    alerts: list[dict],
    governance: pd.DataFrame | None = None,
) -> list[str]:
    """Evaluate alert rules; return one human-readable message per fired alert.

    ``forecasts`` needs ``unique_id`` plus a ``yhat`` column whose first row
    per series is the next-period forecast (the h-step frame from
    :func:`forecast_frame` works as-is). Rules with ``series="*"`` match
    every series in ``panel``. Kinds:

    - ``forecast_below`` fires when a series' next-period forecast is below
      ``threshold``.
    - ``coverage_below`` fires when a series' mean interval coverage — read
      from a precomputed ``governance`` frame (see :func:`governance_frame`)
      — is below ``threshold``; passing such a rule without ``governance``
      raises :class:`ValueError`.
    """
    if not alerts:
        return []
    uids = sorted(panel[ID_COL].astype(str).unique())
    next_forecast = forecasts.groupby(ID_COL, sort=True)["yhat"].first()
    mean_coverage = None
    if governance is not None:
        mean_coverage = governance.groupby(ID_COL, sort=True)["coverage"].mean()
    fired: list[str] = []
    for alert in alerts:
        kind = alert.get("kind")
        series = alert.get("series", "*")
        threshold = float(alert["threshold"])
        matched = uids if series == "*" else [series]
        if kind == "forecast_below":
            for uid in matched:
                value = next_forecast.get(uid)
                if value is not None and float(value) < threshold:
                    fired.append(
                        f"forecast_below: {uid} next-period forecast "
                        f"{float(value):,.1f} < {threshold:,.1f}"
                    )
        elif kind == "coverage_below":
            if mean_coverage is None:
                raise ValueError(
                    "coverage_below alerts need a governance frame; pass "
                    "governance=governance_frame(panel, settings)"
                )
            for uid in matched:
                value = mean_coverage.get(uid)
                if value is not None and float(value) < threshold:
                    fired.append(
                        f"coverage_below: {uid} coverage {float(value):.2f} "
                        f"< {threshold:.2f}"
                    )
        else:
            raise ValueError(
                f"unknown alert kind {kind!r}; expected 'forecast_below' "
                f"or 'coverage_below'"
            )
    return fired
