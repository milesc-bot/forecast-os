"""Tests for the synthetic generators and the embedded AirPassengers dataset."""

import numpy as np
import pandas as pd
import pytest

from forecast_os.core.types import validate_panel
from forecast_os.datasets.air_passengers import load_air_passengers
from forecast_os.datasets.synthetic import generate_returns, generate_series

# -- generate_series ----------------------------------------------------------


def test_generate_series_shape_and_contract():
    df = generate_series(n_series=3, length=40, seed=0)
    assert len(df) == 3 * 40
    assert list(df.columns) == ["unique_id", "ds", "y"]
    assert set(df["unique_id"]) == {"series-0", "series-1", "series-2"}
    assert (df.groupby("unique_id").size() == 40).all()
    validate_panel(df)


def test_generate_series_deterministic_per_seed():
    a = generate_series(n_series=2, length=30, seed=123)
    b = generate_series(n_series=2, length=30, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_generate_series_different_seeds_differ():
    a = generate_series(n_series=1, length=30, seed=1)
    b = generate_series(n_series=1, length=30, seed=2)
    assert not np.allclose(a["y"], b["y"])


def test_generate_series_seasonality_none_is_pure_trend():
    df = generate_series(n_series=2, length=30, seasonality=None, noise=0.0, seed=0)
    for _, g in df.groupby("unique_id"):
        # linear: second differences vanish
        np.testing.assert_allclose(np.diff(g["y"].to_numpy(), n=2), 0.0, atol=1e-9)


def test_generate_series_seasonality_adds_cycle():
    df = generate_series(n_series=1, length=56, seasonality=7, noise=0.0, seed=0)
    y = df["y"].to_numpy()
    assert not np.allclose(np.diff(y, n=2), 0.0, atol=1e-6)
    # trend + exact period-7 cycle: the lag-7 difference is constant (7 * trend)
    lag7 = y[7:] - y[:-7]
    np.testing.assert_allclose(lag7, lag7[0], atol=1e-9)


def test_generate_series_invalid_args_raise():
    with pytest.raises(ValueError):
        generate_series(n_series=0)
    with pytest.raises(ValueError):
        generate_series(length=0)


# -- generate_returns ---------------------------------------------------------


def test_generate_returns_shape_and_determinism():
    a = generate_returns(n_series=2, length=100, seed=9)
    b = generate_returns(n_series=2, length=100, seed=9)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 200
    assert set(a["unique_id"]) == {"asset-0", "asset-1"}
    validate_panel(a)


def test_generate_returns_iid_moments():
    df = generate_returns(n_series=1, length=5000, mu=0.001, sigma=0.02, seed=0)
    y = df["y"].to_numpy()
    assert np.mean(y) == pytest.approx(0.001, abs=0.001)
    assert np.std(y) == pytest.approx(0.02, rel=0.1)


@pytest.mark.parametrize(
    "garch",
    [
        (1e-6, 0.5, 0.5),  # alpha + beta == 1
        (1e-6, 0.6, 0.6),  # alpha + beta > 1
        (-1e-6, 0.1, 0.8),  # negative omega
    ],
)
def test_generate_returns_invalid_garch_raises(garch):
    with pytest.raises(ValueError, match="invalid GARCH"):
        generate_returns(length=50, garch=garch)


def test_generate_returns_valid_garch_runs():
    df = generate_returns(length=300, garch=(2e-6, 0.08, 0.85), seed=13)
    assert len(df) == 300
    assert np.isfinite(df["y"]).all()


# -- AirPassengers ------------------------------------------------------------


def test_air_passengers_values_and_shape():
    df = load_air_passengers()
    assert len(df) == 144
    assert df["y"].iloc[0] == 112.0
    assert df["y"].iloc[-1] == 432.0
    assert df["y"].dtype == float
    assert set(df["unique_id"]) == {"AirPassengers"}
    validate_panel(df)


def test_air_passengers_monotone_monthly_ds():
    df = load_air_passengers()
    assert df["ds"].is_monotonic_increasing
    assert pd.infer_freq(pd.DatetimeIndex(df["ds"])) == "MS"
    assert df["ds"].iloc[0] == pd.Timestamp("1949-01-01")
    assert df["ds"].iloc[-1] == pd.Timestamp("1960-12-01")
