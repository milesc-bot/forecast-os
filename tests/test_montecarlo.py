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


def test_mu_is_the_arithmetic_drift_not_the_log_return_drift():
    """Pin the calibration convention the module docstring promises.

    ``simulate`` applies the Ito correction, ``exp((mu - sigma^2/2) + sigma*Z)``,
    which makes ``mu`` the ARITHMETIC (simple-return) drift: ``E[S_t/S_{t-1}] =
    exp(mu)``, with the mean LOG step equal to ``mu - sigma^2/2``. The module
    docstring used to print that same formula and then conclude "so mu/sigma
    are per-period log-return drift and volatility", which is self-contradictory
    and led callers to calibrate from ``np.diff(np.log(prices))`` — biasing the
    whole fan down by sigma^2/2 per step -- a shortfall of
    ``1 - exp(-sigma^2 * h / 2)``, i.e. ~-1.25% on a one-year daily median at
    1% daily vol. The maths is right; the contract around it was not.
    """
    mu, sigma = 0.001, 0.02
    gross = MonteCarloSimulator(mu=mu, sigma=sigma, seed=0).simulate(100.0, 1, 400_000)[:, 0]
    gross = gross / 100.0
    assert gross.mean() == pytest.approx(np.exp(mu), rel=3e-4)
    assert np.log(gross).mean() == pytest.approx(mu - 0.5 * sigma**2, abs=2e-4)


def test_from_returns_on_simple_returns_is_unbiased_against_the_source():
    """Calibrating from simple returns reproduces the observed mean gross return."""
    rng = np.random.default_rng(3)
    prices = 30.0 * np.exp(np.cumsum(0.0004 + 0.012 * rng.standard_normal(2000)))
    simple = np.diff(prices) / prices[:-1]

    sim = MonteCarloSimulator.from_returns(simple, seed=0)
    gross = sim.simulate(prices[-1], 1, 200_000)[:, 0] / prices[-1]
    assert gross.mean() == pytest.approx(1.0 + simple.mean(), rel=3e-3)

    # the log-return calibration is the biased one, low by ~sigma^2/2 per step
    log_sim = MonteCarloSimulator.from_returns(np.diff(np.log(prices)), seed=0)
    assert log_sim.mu < sim.mu
    assert sim.mu - log_sim.mu == pytest.approx(0.5 * sim.sigma**2, rel=0.15)


def test_log_calibration_shortfall_matches_the_documented_magnitude():
    """Pin the size of the log-return mis-calibration the module docstring quotes.

    The docstring's whole value is its number. It claimed "about -10% on a
    one-year daily fan at 1% daily vol"; the shortfall is
    ``1 - exp(-sigma^2 * h / 2)`` = 1.25% there, ~8x smaller than advertised
    (it takes sigma ~= 2.9% to make the -10% figure true). This test fails if
    the analytic formula and the simulated fan ever disagree with the text.
    """
    sigma, h = 0.01, 252
    analytic = 1.0 - np.exp(-0.5 * sigma**2 * h)
    assert analytic == pytest.approx(0.0125, abs=5e-5)
    assert 1.0 - np.exp(-0.5 * 0.0289**2 * h) == pytest.approx(0.10, abs=5e-3)

    mu = 0.0005
    correct = MonteCarloSimulator(mu=mu, sigma=sigma, seed=0).simulate(100.0, h, 20_000)[:, -1]
    # the same caller feeding LOG returns hands over mu - sigma^2/2 instead
    biased = MonteCarloSimulator(mu=mu - 0.5 * sigma**2, sigma=sigma, seed=0).simulate(
        100.0, h, 20_000
    )[:, -1]
    assert 1.0 - np.median(biased) / np.median(correct) == pytest.approx(analytic, rel=1e-6)
