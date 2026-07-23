"""Financial performance and risk metrics on per-period return arrays.

All functions accept array-likes of simple per-period returns and raise
``ValueError`` on empty input. Ratio metrics (sharpe/sortino/calmar) return
``nan`` when their denominator is ~0; ``rf`` is a per-period risk-free rate.
VaR/CVaR follow the positive-loss convention (a 5% loss -> 0.05).
"""

from __future__ import annotations

import numpy as np

__all__ = [
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

_EPS = 1e-12


def _to_returns(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=float).ravel()
    if r.size == 0:
        raise ValueError("empty returns")
    return r


def _check_level(level: float) -> float:
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    return float(level)


def annualized_return(returns, periods: int = 252) -> float:
    """Geometric (compounded) return annualized to ``periods`` per year."""
    r = _to_returns(returns)
    equity = float(np.prod(1.0 + r))
    if equity <= 0:
        return -1.0
    return float(equity ** (periods / r.size) - 1.0)


def annualized_vol(returns, periods: int = 252) -> float:
    """Sample standard deviation scaled by sqrt(periods); nan for < 2 obs."""
    r = _to_returns(returns)
    if r.size < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * np.sqrt(periods))


def sharpe_ratio(returns, rf: float = 0.0, periods: int = 252) -> float:
    """Annualized mean/std of excess returns; nan when the std is ~0."""
    r = _to_returns(returns) - rf
    if r.size < 2:
        return float("nan")
    sd = float(np.std(r, ddof=1))
    if sd < _EPS:
        return float("nan")
    return float(np.mean(r) / sd * np.sqrt(periods))


def sortino_ratio(returns, rf: float = 0.0, periods: int = 252) -> float:
    """Like Sharpe but with downside deviation; nan when there is no downside."""
    r = _to_returns(returns) - rf
    downside = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)))
    if downside < _EPS:
        return float("nan")
    return float(np.mean(r) / downside * np.sqrt(periods))


def max_drawdown(returns) -> float:
    """Worst peak-to-trough decline of the cumprod equity curve (<= 0)."""
    r = _to_returns(returns)
    equity = np.concatenate([[1.0], np.cumprod(1.0 + r)])  # start counts as a peak
    peaks = np.maximum.accumulate(equity)
    return float(np.min(equity / peaks - 1.0))


def hit_rate(returns) -> float:
    """Fraction of strictly positive returns."""
    r = _to_returns(returns)
    return float(np.mean(r > 0))


def directional_accuracy(y, yhat) -> float:
    """Fraction of periods where forecast and actual agree in sign."""
    y = np.asarray(y, dtype=float).ravel()
    yhat = np.asarray(yhat, dtype=float).ravel()
    if y.shape != yhat.shape:
        raise ValueError(f"shape mismatch: y {y.shape} vs yhat {yhat.shape}")
    if y.size == 0:
        raise ValueError("empty input")
    return float(np.mean(np.sign(y) == np.sign(yhat)))


def value_at_risk(returns, level: float = 0.95) -> float:
    """Historic VaR at ``level``: the (1-level) return quantile as a loss (>= 0)."""
    r = _to_returns(returns)
    level = _check_level(level)
    return float(max(0.0, -np.quantile(r, 1.0 - level)))


def conditional_var(returns, level: float = 0.95) -> float:
    """Expected shortfall: mean loss over the (1-level) tail (>= 0)."""
    r = _to_returns(returns)
    level = _check_level(level)
    q = float(np.quantile(r, 1.0 - level))
    tail = r[r <= q + _EPS]  # tolerance keeps boundary points in the tail
    return float(max(0.0, -np.mean(tail)))


def calmar_ratio(returns, periods: int = 252) -> float:
    """Annualized return over absolute max drawdown; nan when drawdown is ~0."""
    r = _to_returns(returns)
    mdd = abs(max_drawdown(r))
    if mdd < _EPS:
        return float("nan")
    return annualized_return(r, periods) / mdd
