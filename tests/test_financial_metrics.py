"""Hand-computed cases for finance/metrics.py."""

from __future__ import annotations

import warnings

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


def test_var_and_cvar_propagate_nan_instead_of_reporting_zero_risk():
    """A NaN in the returns must not be reported as "no downside risk".

    Regression: ``value_at_risk`` and ``conditional_var`` ended with
    ``max(0.0, x)``. When ``np.quantile`` returned ``nan`` (any NaN in the
    input), ``nan > 0.0`` is ``False`` so ``max`` yielded ``0.0`` — a risk
    metric silently reporting zero risk on unclean data, which is the worst
    possible direction to fail in. In ``conditional_var`` the tail mask
    ``r <= nan`` was additionally all-False, so ``np.mean`` on the empty slice
    emitted two RuntimeWarnings before being floored to 0.0.

    Correct behaviour is to propagate ``nan``, matching sharpe/sortino/
    annualized_*/max_drawdown/calmar on the same array, and to emit no warning.
    """
    r = [-0.30, -0.10, 0.01, 0.02, 0.03, 0.02, 0.01, 0.00, 0.02, float("nan")]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert np.isnan(value_at_risk(r))
        assert np.isnan(conditional_var(r))
    # siblings agree
    for fn in (sharpe_ratio, sortino_ratio, annualized_vol, max_drawdown, calmar_ratio):
        assert np.isnan(fn(r))
    # dropping the NaN recovers the real (non-zero) risk
    clean = r[:-1]
    assert np.isclose(value_at_risk(clean), 0.22, atol=1e-9)
    assert np.isclose(conditional_var(clean), 0.30, atol=1e-9)


def test_var_and_cvar_still_floor_finite_negatives_at_zero():
    """The nan guard must not disturb the documented >= 0 floor."""
    assert value_at_risk([0.01, 0.02, 0.03]) == 0.0
    assert conditional_var([0.01, 0.02, 0.03]) == 0.0


def test_conditional_var_is_scale_equivariant():
    """Expected shortfall is positively homogeneous: CVaR(c*r) == c*CVaR(r).

    Regression: the tail mask used an absolute tolerance of 1e-12, so for
    returns whose magnitude was near or below 1e-12 the tolerance swept extra
    points into the tail. ``conditional_var(big * 1e-18)`` returned 6.667e-13
    where scale equivariance requires 1e-12 (a 33% error). The tolerance only
    exists to keep floating-point boundary points in the tail, which is an
    inherently relative notion, so it must scale with the data.
    """
    big = np.array([-1e6, -1e6, 0.0] + [1e6] * 17)
    base = conditional_var(big, 0.95)
    assert np.isclose(base, 1e6)
    for scale in (1.0, 1e-3, 1e-6, 1e-9, 1e-18, 1e6):
        # atol=0: the 1e-18 case is ~1e-12, far below np.isclose's default atol
        assert np.isclose(conditional_var(big * scale, 0.95), base * scale, rtol=1e-12, atol=0.0)


def test_conditional_var_keeps_boundary_points_in_the_tail():
    """The relative tolerance must still do the job it was added for."""
    # 5% quantile of 21 points is exactly the 2nd sorted element; both it and
    # the minimum belong to the tail.
    r = np.linspace(-0.10, 0.10, 21)
    assert np.isclose(conditional_var(r, level=0.95), 0.095, atol=1e-9)


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
