"""Forecast-driven strategy backtesting.

``StrategyBacktester`` turns any registered forecaster into a trading rule on
a returns panel: walk-forward one-step forecasts via
:func:`~forecast_os.evaluation.backtest.cross_validation`, positions sized by
one of three rules (binary long/flat, interval-proportional, or fractional
Kelly), and ``cost_bps`` basis points charged per unit of position change.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from scipy import stats

from ..core.base import BaseForecaster
from ..core.types import ID_COL, TARGET_COL, TIME_COL, infer_step, validate_panel
from ..evaluation.backtest import cross_validation
from .metrics import (
    annualized_return,
    annualized_vol,
    calmar_ratio,
    conditional_var,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)

__all__ = ["BacktestResult", "StrategyBacktester"]

_META_COLS = {ID_COL, TIME_COL, TARGET_COL, "cutoff"}
_SIZINGS = ("binary", "proportional", "kelly")
_SIGMA_EPS = 1e-12

#: Periods per year implied by the *base unit* of an inferred pandas frequency
#: string; a multi-unit step like ``"4h"`` divides its unit's rate by the
#: multiple. Daily and business-daily both map to the 252 trading-day
#: convention, which is also the fallback for any step this table does not
#: cover (numeric ``ds``, irregular datetimes, sub-hourly units), preserving the
#: historical default.
_PERIODS_BY_FREQ = {
    "D": 252,
    "B": 252,
    "W": 52,
    "M": 12,
    "MS": 12,
    "ME": 12,
    "Q": 4,
    "QS": 4,
    "QE": 4,
    "Y": 1,
    "YS": 1,
    "YE": 1,
    "A": 1,
    "h": 24 * 252,
    "H": 24 * 252,
    # Sub-hourly, on the same 24*252 trading-year convention as "h". Without
    # these a 15-minute panel fell back to 252 and understated annualized vol
    # ~10x; a 1-minute panel ~38x.
    "min": 24 * 252 * 60,
    "T": 24 * 252 * 60,
    "s": 24 * 252 * 3600,
    "S": 24 * 252 * 3600,
    # Business-anchored month/quarter/year ends. These were the worse case:
    # falling back to 252 OVERSTATED a BME panel's Sharpe 4.6x and a BYE
    # panel's 15.9x, because the true rate is 12 and 1.
    "BME": 12, "BMS": 12, "BM": 12,
    "BQE": 4, "BQS": 4, "BQ": 4,
    "BYE": 1, "BYS": 1, "BA": 1, "BY": 1,
}
_DEFAULT_PERIODS = 252


def _infer_periods(df: pd.DataFrame) -> int:
    """Periods per year implied by the panel's modal per-series time step.

    The step's multiple counts: ``"4h"`` bars occur a quarter as often as
    ``"h"`` bars, so the base unit's rate is divided by ``to_offset(freq).n``.

    Expects a frame that has already been through
    :func:`~forecast_os.core.types.validate_panel`: ``ds`` may legally arrive
    as ISO strings or ``datetime.date`` objects and only validation normalises
    those to datetime64.
    """
    steps = [infer_step(g[TIME_COL]) for _, g in df.groupby(ID_COL, sort=True)]
    if not steps:
        return _DEFAULT_PERIODS
    modal = Counter(steps).most_common(1)[0][0]
    if not isinstance(modal, str):
        return _DEFAULT_PERIODS
    try:
        offset = to_offset(modal)
    except ValueError:  # pragma: no cover - infer_freq only returns parseable strings
        return _PERIODS_BY_FREQ.get(modal.split("-")[0], _DEFAULT_PERIODS)
    base = _PERIODS_BY_FREQ.get(offset.name.split("-")[0])
    if base is None:
        return _DEFAULT_PERIODS
    return max(1, round(base / (abs(offset.n) or 1)))


@dataclass
class BacktestResult:
    """Strategy backtest output.

    ``summary`` has one row per ``unique_id`` (total_return, annualized_return,
    sharpe, sortino, max_drawdown, hit_rate, n_trades, exposure,
    annualized_vol, calmar, var_95, cvar_95); ``frame`` is the per-period
    detail (unique_id, ds, y, yhat, position, strategy_return, equity). When
    ``sizing != "binary"`` the frame also carries a ``sigma`` column (the
    per-step forecast standard deviation recovered from the model's prediction
    interval, inserted after ``yhat``); for binary sizing the column is
    omitted entirely.

    ``periods`` is the periods-per-year actually used to annualize
    annualized_return, sharpe, sortino, annualized_vol and calmar — either the
    value passed to :class:`StrategyBacktester` or the one inferred from the
    panel's time step and the run's ``step_size`` (the realized returns are
    ``step_size`` bars apart, so the panel's rate is divided by it).
    """

    summary: pd.DataFrame
    frame: pd.DataFrame
    periods: int = _DEFAULT_PERIODS


class StrategyBacktester:
    """Backtest a forecaster as a trading rule on a (unique_id, ds, y) returns panel.

    Position sizing rules (``sizing``):

    - ``"binary"`` (default): long/flat — position is ``1.0`` when
      ``yhat > threshold`` and ``0.0`` otherwise.
    - ``"proportional"``: confidence-scaled long/flat in ``[0, 1]``. With
      ``P = P(r > threshold | r ~ N(yhat, sigma)) = 1 - Phi((threshold - yhat) / sigma)``
      the position is ``clip(2 * (P - 0.5), 0, 1)`` — zero at or below the
      threshold, approaching 1 as the forecast edge grows vs. its uncertainty.
    - ``"kelly"``: fractional-Kelly long-only sizing,
      ``clip(kelly_fraction * (yhat - threshold) / sigma**2, 0, max_leverage)``.

    ``"proportional"`` and ``"kelly"`` consume the model's prediction
    intervals: cross-validation is run with ``level=[level]`` and the per-step
    forecast standard deviation is recovered as ``sigma = (hi - lo) / (2 * z)``
    with ``z = norm.ppf(0.5 + level / 200)``. Steps whose ``sigma`` is
    non-finite or ``<= 1e-12`` fall back to the binary rule.

    Trading costs of ``cost_bps / 1e4`` per unit of position change are
    charged on entry and on every resize (``|position_t - position_{t-1}|``,
    starting flat).

    ``periods`` is the periods-per-year used to annualize annualized_return,
    sharpe, sortino, annualized_vol and calmar. Left as ``None`` it is inferred
    from the (validated) panel's modal time step (daily/business-daily 252,
    weekly 52, monthly 12, quarterly 4, yearly 1, and 24*252 per hour down to
    seconds, business-anchored month/quarter/year ends included), divided by the
    step's multiple where it has one (``"4h"`` -> 1512, ``"2W"`` -> 26) and
    rounded to at least 1 — so steps coarser than a year (``"2YE"``, ``"3QS"``)
    all annualize as 1. It falls back to 252 for numeric ``ds``, irregular
    datetimes and any unit outside that list, and is then divided by
    :meth:`run`'s ``step_size`` so it describes the strategy's own return series
    rather than the panel's bar spacing. The resolved value is reported on
    :attr:`BacktestResult.periods`; pass ``periods`` explicitly whenever the
    inferred convention does not match your bars.
    """

    def __init__(
        self,
        model: BaseForecaster | str,
        threshold: float = 0.0,
        cost_bps: float = 0.0,
        sizing: str = "binary",
        level: int = 80,
        kelly_fraction: float = 0.5,
        max_leverage: float = 1.0,
        periods: int | None = None,
    ):
        if sizing not in _SIZINGS:
            raise ValueError(f"sizing must be one of {_SIZINGS}, got {sizing!r}")
        if not (0 < level < 100) or int(level) != level:
            raise ValueError(f"level must be an integer in (0, 100), got {level!r}")
        if not np.isfinite(kelly_fraction) or kelly_fraction <= 0:
            raise ValueError(f"kelly_fraction must be finite and > 0, got {kelly_fraction!r}")
        if not np.isfinite(max_leverage) or max_leverage <= 0:
            raise ValueError(f"max_leverage must be finite and > 0, got {max_leverage!r}")
        if periods is not None and (int(periods) != periods or periods < 1):
            raise ValueError(f"periods must be None or an integer >= 1, got {periods!r}")
        self.periods = None if periods is None else int(periods)
        self.model = model
        self.threshold = float(threshold)
        self.cost_bps = float(cost_bps)
        self.sizing = sizing
        self.level = int(level)
        self.kelly_fraction = float(kelly_fraction)
        self.max_leverage = float(max_leverage)

    def _sized_positions(
        self, yhat: np.ndarray, sigma: np.ndarray, binary: np.ndarray
    ) -> np.ndarray:
        """Apply the proportional/kelly rule; degenerate-sigma steps go binary."""
        valid = np.isfinite(sigma) & (sigma > _SIGMA_EPS)
        safe = np.where(valid, sigma, 1.0)  # dummy divisor; invalid steps replaced below
        if self.sizing == "proportional":
            p_up = stats.norm.sf((self.threshold - yhat) / safe)
            pos = np.clip(2.0 * (p_up - 0.5), 0.0, 1.0)
        else:  # kelly
            pos = np.clip(
                self.kelly_fraction * (yhat - self.threshold) / safe**2,
                0.0,
                self.max_leverage,
            )
        fallback = binary
        if self.sizing == "kelly":
            # the binary fallback must still honor the hard leverage cap
            fallback = np.minimum(binary, self.max_leverage)
        return np.where(valid, pos, fallback)

    def run(self, df: pd.DataFrame, test_size: int = 60, step_size: int = 1) -> BacktestResult:
        """Walk-forward backtest over the last ``test_size`` one-step windows."""
        df = validate_panel(df)  # normalises ds before any inference reads it
        level = None if self.sizing == "binary" else [self.level]
        cv = cross_validation(
            df, [self.model], h=1, n_windows=test_size, step_size=step_size, level=level
        )
        if self.periods is not None:
            periods = self.periods
        else:
            # Consecutive realized returns are step_size bars apart, so the
            # panel's own periods-per-year has to be divided by that cadence.
            periods = max(1, round(_infer_periods(df) / step_size))
        forecast_col = next(
            c for c in cv.columns if c not in _META_COLS and "-lo-" not in c and "-hi-" not in c
        )
        z = float(stats.norm.ppf(0.5 + self.level / 200))
        frames, rows = [], []
        for uid, g in cv.groupby(ID_COL, sort=True):
            g = g.sort_values(TIME_COL, kind="stable")
            y = g[TARGET_COL].to_numpy(dtype=float)
            yhat = g[forecast_col].to_numpy(dtype=float)
            binary = (yhat > self.threshold).astype(float)
            if self.sizing == "binary":
                sigma = None
                position = binary
            else:
                lo = g[f"{forecast_col}-lo-{self.level}"].to_numpy(dtype=float)
                hi = g[f"{forecast_col}-hi-{self.level}"].to_numpy(dtype=float)
                sigma = (hi - lo) / (2.0 * z)
                position = self._sized_positions(yhat, sigma, binary)
            dpos = np.abs(np.diff(np.concatenate([[0.0], position])))  # flat before start
            strat_ret = position * y - self.cost_bps / 1e4 * dpos
            equity = np.cumprod(1.0 + strat_ret)
            data = {
                ID_COL: uid,
                TIME_COL: g[TIME_COL].to_numpy(),
                "y": y,
                "yhat": yhat,
                "position": position,
                "strategy_return": strat_ret,
                "equity": equity,
            }
            frame = pd.DataFrame(data)
            if sigma is not None:
                frame.insert(4, "sigma", sigma)
            frames.append(frame)
            rows.append(
                {
                    ID_COL: uid,
                    "total_return": float(equity[-1] - 1.0),
                    "annualized_return": annualized_return(strat_ret, periods=periods),
                    "sharpe": sharpe_ratio(strat_ret, periods=periods),
                    "sortino": sortino_ratio(strat_ret, periods=periods),
                    "max_drawdown": max_drawdown(strat_ret),
                    "hit_rate": hit_rate(strat_ret),
                    "n_trades": int(np.sum(dpos > 0)),
                    "exposure": float(np.mean(position)),
                    "annualized_vol": annualized_vol(strat_ret, periods=periods),
                    "calmar": calmar_ratio(strat_ret, periods=periods),
                    "var_95": value_at_risk(strat_ret, level=0.95),
                    "cvar_95": conditional_var(strat_ret, level=0.95),
                }
            )
        return BacktestResult(
            summary=pd.DataFrame(rows),
            frame=pd.concat(frames, ignore_index=True),
            periods=periods,
        )
