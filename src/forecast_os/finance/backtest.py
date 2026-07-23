"""Forecast-driven strategy backtesting.

``StrategyBacktester`` turns any registered forecaster into a long/flat
trading rule on a returns panel: walk-forward one-step forecasts via
:func:`~forecast_os.evaluation.backtest.cross_validation`, go long whenever
the forecast exceeds ``threshold``, and charge ``cost_bps`` basis points per
unit of position change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.base import BaseForecaster
from ..core.types import ID_COL, TARGET_COL, TIME_COL
from ..evaluation.backtest import cross_validation
from .metrics import annualized_return, hit_rate, max_drawdown, sharpe_ratio, sortino_ratio

__all__ = ["BacktestResult", "StrategyBacktester"]

_META_COLS = {ID_COL, TIME_COL, TARGET_COL, "cutoff"}


@dataclass
class BacktestResult:
    """Strategy backtest output.

    ``summary`` has one row per ``unique_id`` (total_return, annualized_return,
    sharpe, sortino, max_drawdown, hit_rate, n_trades, exposure); ``frame`` is
    the per-period detail (unique_id, ds, y, yhat, position, strategy_return,
    equity).
    """

    summary: pd.DataFrame
    frame: pd.DataFrame


class StrategyBacktester:
    """Long/flat backtest of a forecaster on a (unique_id, ds, y) returns panel."""

    def __init__(
        self,
        model: BaseForecaster | str,
        threshold: float = 0.0,
        cost_bps: float = 0.0,
    ):
        self.model = model
        self.threshold = float(threshold)
        self.cost_bps = float(cost_bps)

    def run(self, df: pd.DataFrame, test_size: int = 60, step_size: int = 1) -> BacktestResult:
        """Walk-forward backtest over the last ``test_size`` one-step windows."""
        cv = cross_validation(
            df, [self.model], h=1, n_windows=test_size, step_size=step_size
        )
        forecast_col = next(
            c for c in cv.columns if c not in _META_COLS and "-lo-" not in c and "-hi-" not in c
        )
        frames, rows = [], []
        for uid, g in cv.groupby(ID_COL, sort=True):
            g = g.sort_values(TIME_COL, kind="stable")
            y = g[TARGET_COL].to_numpy(dtype=float)
            yhat = g[forecast_col].to_numpy(dtype=float)
            position = (yhat > self.threshold).astype(float)
            dpos = np.abs(np.diff(np.concatenate([[0.0], position])))  # flat before start
            strat_ret = position * y - self.cost_bps / 1e4 * dpos
            equity = np.cumprod(1.0 + strat_ret)
            frames.append(
                pd.DataFrame(
                    {
                        ID_COL: uid,
                        TIME_COL: g[TIME_COL].to_numpy(),
                        "y": y,
                        "yhat": yhat,
                        "position": position,
                        "strategy_return": strat_ret,
                        "equity": equity,
                    }
                )
            )
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
                }
            )
        return BacktestResult(
            summary=pd.DataFrame(rows), frame=pd.concat(frames, ignore_index=True)
        )
