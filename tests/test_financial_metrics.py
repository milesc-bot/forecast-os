"""Hand-computed cases for finance/metrics.py."""

from __future__ import annotations

import numpy as np
import pytest

from forecast_os.finance.metrics import (
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


def test_annualized_return_geometric():
    # 252 periods of 1% compound to exactly one "year"
    assert np.isclose(annualized_return([0.01] * 252), 1.01**252 - 1)
    # a single 5% period annualized over 12 periods/year
    assert np.isclose(annualized_return([0.05], periods=12), 1.05**12 - 1)
    assert annualized_return([0.0, 0.0]) == 0.0


def test_annualized_vol():
    r = [0.01, -0.01, 0.01, -0.01]
    assert np.isclose(annualized_vol(r), np.std(r, ddof=1) * np.sqrt(252))
    assert np.isnan(annualized_vol([0.01]))


def test_sharpe_positive_drift_and_nan_on_constant():
    r = [0.02, 0.0] * 50
    expected = 0.01 / np.std(r, ddof=1) / 1.0 * np.sqrt(252)
    assert np.isclose(sharpe_ratio(r), expected)
    assert sharpe_ratio(r) > 0
    # per-period risk-free rate reduces the ratio
    assert sharpe_ratio(r, rf=0.005) < sharpe_ratio(r)
    # zero-variance returns: denominator ~0 -> nan
    assert np.isnan(sharpe_ratio([0.01] * 100))


def test_sortino():
    r = [0.02, -0.01]
    downside = np.sqrt(np.mean([0.0, 0.01**2]))
    assert np.isclose(sortino_ratio(r), 0.005 / downside * np.sqrt(252))
    # no downside moves -> nan
    assert np.isnan(sortino_ratio([0.01, 0.02, 0.03]))


def test_max_drawdown():
    assert np.isclose(max_drawdown([0.1, -0.5, 0.2]), -0.5)
    assert max_drawdown([0.01] * 50) == 0.0
    # a first-period loss is a drawdown from the starting capital
    assert np.isclose(max_drawdown([-0.1, 0.05]), -0.1)
    assert max_drawdown(np.random.default_rng(0).normal(0, 0.01, 200)) <= 0.0


def test_hit_rate():
    r = [0.01, -0.01] * 50
    assert hit_rate(r) == 0.5
    assert hit_rate([0.01, 0.02]) == 1.0
    assert hit_rate([-0.01, 0.0]) == 0.0  # zeros are not hits


def test_directional_accuracy():
    y = [1.0, -1.0, 1.0, -1.0]
    yhat = [2.0, -3.0, -1.0, -2.0]
    assert directional_accuracy(y, yhat) == 0.75
    assert directional_accuracy([1.0, 2.0], [3.0, 4.0]) == 1.0
    with pytest.raises(ValueError):
        directional_accuracy([1.0, 2.0], [1.0])


def test_value_at_risk_and_cvar_known_array():
    r = np.linspace(-0.10, 0.10, 21)  # -10%, -9%, ..., +10%
    # 5% quantile of 21 sorted points is exactly the 2nd element (-0.09)
    assert np.isclose(value_at_risk(r, level=0.95), 0.09, atol=1e-6)
    # CVaR = -mean of the tail {-0.10, -0.09}
    assert np.isclose(conditional_var(r, level=0.95), 0.095, atol=1e-6)


def test_value_at_risk_floor_and_validation():
    assert value_at_risk([0.01, 0.02, 0.03]) == 0.0  # all gains: VaR floored at 0
    assert value_at_risk([-0.05, -0.02, 0.01, 0.03]) >= 0.0
    with pytest.raises(ValueError):
        value_at_risk([0.01], level=1.5)
    with pytest.raises(ValueError):
        conditional_var([0.01], level=0.0)


def test_calmar():
    r = [0.1, -0.5, 0.2]
    assert np.isclose(calmar_ratio(r), annualized_return(r) / 0.5)
    assert np.isnan(calmar_ratio([0.01] * 50))  # no drawdown -> nan


@pytest.mark.parametrize(
    "fn",
    [
        annualized_return,
        annualized_vol,
        sharpe_ratio,
        sortino_ratio,
        max_drawdown,
        hit_rate,
        value_at_risk,
        conditional_var,
        calmar_ratio,
    ],
)
def test_empty_raises(fn):
    with pytest.raises(ValueError):
        fn([])


def test_accepts_array_likes():
    for arr in ([0.01, -0.02, 0.03], np.array([0.01, -0.02, 0.03])):
        assert np.isfinite(max_drawdown(arr))
        assert np.isfinite(hit_rate(arr))
