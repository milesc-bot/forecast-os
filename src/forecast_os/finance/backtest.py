"""Forecast-driven strategy backtesting.

``StrategyBacktester`` turns any registered forecaster into a trading rule on
a returns panel: walk-forward one-step forecasts via
:func:`~forecast_os.evaluation.backtest.cross_validation`, positions sized by
one of three rules (binary long/flat, interval-proportional, or fractional
Kelly), and ``cost_bps`` basis points charged per unit of position change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..core.base import BaseForecaster
from ..core.types import ID_COL, TARGET_COL, TIME_COL
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
    """

    summary: pd.DataFrame
    frame: pd.DataFrame


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
    ):
        if sizing not in _SIZINGS:
            raise ValueError(f"sizing must be one of {_SIZINGS}, got {sizing!r}")
        if not (0 < level < 100) or int(level) != level:
            raise ValueError(f"level must be an integer in (0, 100), got {level!r}")
        if not np.isfinite(kelly_fraction) or kelly_fraction <= 0:
            raise ValueError(f"kelly_fraction must be finite and > 0, got {kelly_fraction!r}")
        if not np.isfinite(max_leverage) or max_leverage <= 0:
            raise ValueError(f"max_leverage must be finite and > 0, got {max_leverage!r}")
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
        return np.where(valid, pos, binary)

    def run(self, df: pd.DataFrame, test_size: int = 60, step_size: int = 1) -> BacktestResult:
        """Walk-forward backtest over the last ``test_size`` one-step windows."""
        level = None if self.sizing == "binary" else [self.level]
        cv = cross_validation(
            df, [self.model], h=1, n_windows=test_size, step_size=step_size, level=level
        )
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
                    "annualized_return": annualized_return(strat_ret),
                    "sharpe": sharpe_ratio(strat_ret),
                    "sortino": sortino_ratio(strat_ret),
                    "max_drawdown": max_drawdown(strat_ret),
                    "hit_rate": hit_rate(strat_ret),
                    "n_trades": int(np.sum(dpos > 0)),
                    "exposure": float(np.mean(position)),
                    "annualized_vol": annualized_vol(strat_ret),
                    "calmar": calmar_ratio(strat_ret),
                    "var_95": value_at_risk(strat_ret, level=0.95),
                    "cvar_95": conditional_var(strat_ret, level=0.95),
                }
            )
        return BacktestResult(
            summary=pd.DataFrame(rows), frame=pd.concat(frames, ignore_index=True)
        )
