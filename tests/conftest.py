"""Shared fixtures. All synthetic data is seeded for determinism."""

import numpy as np
import pytest

from forecast_os.datasets.synthetic import generate_returns, generate_series


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def panel():
    """3 daily series, 200 obs: trend + weekly seasonality + light noise."""
    return generate_series(
        n_series=3, length=200, freq="D", trend=0.5, seasonality=7, season_amp=10.0,
        noise=1.0, seed=7,
    )


@pytest.fixture
def trend_panel():
    """2 series with a clean trend and no seasonality."""
    return generate_series(
        n_series=2, length=120, freq="D", trend=1.0, seasonality=None, noise=0.5, seed=11,
    )


@pytest.fixture
def flat_panel():
    """2 series of pure noise around a level (no trend/seasonality)."""
    return generate_series(
        n_series=2, length=150, freq="D", trend=0.0, seasonality=None, noise=2.0, seed=3,
    )


@pytest.fixture
def returns_panel():
    """1 asset, 600 business-day i.i.d. Gaussian returns."""
    return generate_returns(n_series=1, length=600, mu=0.0004, sigma=0.012, seed=5)


@pytest.fixture
def garch_returns_panel():
    """1 asset, 1200 returns from a GARCH(1,1) process (persistence 0.93)."""
    return generate_returns(
        n_series=1, length=1200, mu=0.0, garch=(2e-6, 0.08, 0.85), seed=13,
    )
