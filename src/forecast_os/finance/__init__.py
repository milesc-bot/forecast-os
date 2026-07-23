"""Quantitative finance layer: volatility, simulation, regimes, risk, backtesting."""

from .backtest import BacktestResult, StrategyBacktester
from .garch import GARCH11, GARCHVolatility
from .metrics import (
    annualized_return,
    annualized_vol,
    calmar_ratio,
    conditional_var,
    directional_accuracy,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)
from .montecarlo import MonteCarloSimulator
from .regime import MarkovRegimeSwitching

__all__ = [
    "GARCH11",
    "GARCHVolatility",
    "MonteCarloSimulator",
    "MarkovRegimeSwitching",
    "StrategyBacktester",
    "BacktestResult",
    "annualized_return",
    "annualized_vol",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "hit_rate",
    "directional_accuracy",
    "value_at_risk",
    "conditional_var",
    "calmar_ratio",
]
