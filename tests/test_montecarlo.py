"""Tests for finance/montecarlo.py: GBM MonteCarloSimulator."""

from __future__ import annotations

import numpy as np
import pytest

from forecast_os.finance.montecarlo import MonteCarloSimulator


def test_simulate_shape_and_positive():
    sim = MonteCarloSimulator(mu=0.001, sigma=0.02, seed=1)
    paths = sim.simulate(100.0, 12, n_paths=50)
    assert paths.shape == (50, 12)
    assert np.isfinite(paths).all()
    assert (paths > 0).all()


def test_near_zero_sigma_is_deterministic_drift():
    sim = MonteCarloSimulator(mu=0.001, sigma=1e-9, seed=0)
    paths = sim.simulate(100.0, 10, n_paths=20)
    k = np.arange(1, 11)
    expected = 100.0 * np.exp(0.001 * k)
    np.testing.assert_allclose(paths, np.tile(expected, (20, 1)), rtol=1e-5)


def test_seed_reproducibility():
    a = MonteCarloSimulator(mu=0.0, sigma=0.02, seed=7).simulate(50.0, 8, 30)
    b = MonteCarloSimulator(mu=0.0, sigma=0.02, seed=7).simulate(50.0, 8, 30)
    c = MonteCarloSimulator(mu=0.0, sigma=0.02, seed=8).simulate(50.0, 8, 30)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    # repeated calls on the same simulator are also reproducible
    sim = MonteCarloSimulator(mu=0.0, sigma=0.02, seed=7)
    np.testing.assert_array_equal(sim.simulate(50.0, 8, 30), sim.simulate(50.0, 8, 30))


def test_summary_shape_and_columns():
    sim = MonteCarloSimulator(mu=0.0005, sigma=0.02, seed=3)
    out = sim.summary(100.0, 15, n_paths=500)
    assert list(out.columns) == ["step", "q05", "q25", "q50", "q75", "q95"]
    assert len(out) == 15
    np.testing.assert_array_equal(out["step"].to_numpy(), np.arange(1, 16))


def test_summary_custom_levels():
    sim = MonteCarloSimulator(seed=3)
    out = sim.summary(100.0, 5, n_paths=200, levels=(10, 90))
    assert list(out.columns) == ["step", "q10", "q90"]


def test_quantile_monotonicity():
    sim = MonteCarloSimulator(mu=0.0, sigma=0.02, seed=11)
    out = sim.summary(100.0, 20, n_paths=2000)
    assert (out["q05"] < out["q50"]).all()
    assert (out["q50"] < out["q95"]).all()
    assert (out["q25"] < out["q75"]).all()


def test_from_returns_estimates_mu_sigma():
    rng = np.random.default_rng(5)
    r = 0.001 + 0.015 * rng.standard_normal(400)
    sim = MonteCarloSimulator.from_returns(r)
    assert isinstance(sim, MonteCarloSimulator)
    assert np.isclose(sim.mu, np.mean(r))
    assert np.isclose(sim.sigma, np.std(r, ddof=1))


def test_params_stored_as_attributes():
    sim = MonteCarloSimulator(mu=0.002, sigma=0.03, seed=9)
    assert sim.mu == 0.002
    assert sim.sigma == 0.03
    assert sim.seed == 9


def test_invalid_inputs_raise():
    sim = MonteCarloSimulator()
    with pytest.raises(ValueError):
        sim.simulate(100.0, 0)
    with pytest.raises(ValueError):
        sim.simulate(100.0, 5, n_paths=0)
    with pytest.raises(ValueError):
        sim.simulate(-1.0, 5)
    with pytest.raises(ValueError):
        MonteCarloSimulator.from_returns([])
    with pytest.raises(ValueError):
        MonteCarloSimulator(sigma=-0.1)
