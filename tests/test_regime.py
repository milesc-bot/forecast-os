"""Tests for finance/regime.py: Gaussian HMM regime switching fitted by EM."""

from __future__ import annotations

import numpy as np
import pytest

from forecast_os.core.exceptions import NotFittedError
from forecast_os.finance.regime import MarkovRegimeSwitching

BULL_MU, BULL_SIGMA = 0.002, 0.005
BEAR_MU, BEAR_SIGMA = -0.002, 0.02


def _planted():
    """600 bull then 600 bear observations; truth uses mean-ascending state order."""
    rng = np.random.default_rng(0)
    bull = BULL_MU + BULL_SIGMA * rng.standard_normal(600)
    bear = BEAR_MU + BEAR_SIGMA * rng.standard_normal(600)
    r = np.concatenate([bull, bear])
    truth = np.concatenate([np.ones(600, dtype=int), np.zeros(600, dtype=int)])
    return r, truth


@pytest.fixture(scope="module")
def fitted():
    r, truth = _planted()
    model = MarkovRegimeSwitching(n_states=2, seed=0)
    assert model.fit(r) is model
    return model, truth


def test_attribute_shapes(fitted):
    model, _ = fitted
    assert model.means_.shape == (2,)
    assert model.stds_.shape == (2,)
    assert model.transition_.shape == (2, 2)
    assert model.smoothed_probs_.shape == (1200, 2)
    assert model.regimes_.shape == (1200,)
    assert (model.stds_ > 0).all()


def test_states_ordered_by_mean_ascending(fitted):
    model, _ = fitted
    assert model.means_[0] < model.means_[1]
    # state 0 is the bear regime: negative mean, higher volatility
    assert model.means_[0] < -0.0005
    assert model.means_[1] > 0.0005
    assert model.stds_[0] > model.stds_[1]


def test_recovered_moments_close_to_truth(fitted):
    model, _ = fitted
    assert abs(model.means_[0] - BEAR_MU) < 0.002
    assert abs(model.means_[1] - BULL_MU) < 0.002
    assert 0.5 * BEAR_SIGMA < model.stds_[0] < 2.0 * BEAR_SIGMA
    assert 0.5 * BULL_SIGMA < model.stds_[1] < 2.0 * BULL_SIGMA


def test_transition_rows_sum_to_one(fitted):
    model, _ = fitted
    np.testing.assert_allclose(model.transition_.sum(axis=1), [1.0, 1.0], atol=1e-8)
    assert (model.transition_ >= 0).all()


def test_smoothed_probs_valid_and_accurate(fitted):
    model, truth = fitted
    probs = model.smoothed_probs_
    assert ((probs >= 0) & (probs <= 1)).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-8)
    accuracy = float(np.mean(model.regimes_ == truth))
    assert accuracy > 0.8, f"segment accuracy {accuracy:.3f} <= 0.8"


def test_predict_proba(fitted):
    model, _ = fitted
    p1 = model.predict_proba(1)
    assert p1.shape == (2,)
    assert (p1 >= 0).all()
    assert np.isclose(p1.sum(), 1.0)
    np.testing.assert_allclose(p1, model.smoothed_probs_[-1] @ model.transition_)
    p10 = model.predict_proba(10)
    expected = model.smoothed_probs_[-1] @ np.linalg.matrix_power(model.transition_, 10)
    np.testing.assert_allclose(p10, expected)


def test_expected_return(fitted):
    model, _ = fitted
    er = model.expected_return(5)
    assert isinstance(er, float)
    assert np.isfinite(er)
    assert np.isclose(er, float(model.predict_proba(5) @ model.means_))


def test_deterministic_given_seed():
    r, _ = _planted()
    m1 = MarkovRegimeSwitching(n_states=2, seed=0).fit(r)
    m2 = MarkovRegimeSwitching(n_states=2, seed=0).fit(r)
    np.testing.assert_allclose(m1.means_, m2.means_)
    np.testing.assert_allclose(m1.transition_, m2.transition_)


def test_params_stored_as_attributes():
    model = MarkovRegimeSwitching(n_states=3, max_iter=50, tol=1e-4, seed=2)
    assert model.n_states == 3
    assert model.max_iter == 50
    assert model.tol == 1e-4
    assert model.seed == 2


def test_errors():
    with pytest.raises(ValueError):
        MarkovRegimeSwitching().fit(np.array([]))
    with pytest.raises(ValueError):
        MarkovRegimeSwitching().fit(np.array([0.01, 0.02]))  # too short
    with pytest.raises(NotFittedError):
        MarkovRegimeSwitching().predict_proba(3)
    r, _ = _planted()
    model = MarkovRegimeSwitching().fit(r)
    with pytest.raises(ValueError):
        model.predict_proba(0)
