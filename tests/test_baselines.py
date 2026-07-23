"""Tests for the baseline forecasters: naive, seasonal_naive, drift, window_average."""

import numpy as np
import pytest

from forecast_os.core.exceptions import ForecastOSError, NotFittedError
from forecast_os.core.registry import _REGISTRY
from forecast_os.core.types import to_panel
from forecast_os.models.baselines import Drift, Naive, SeasonalNaive, WindowAverage

Y6 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


# -- exact arithmetic ---------------------------------------------------------


def test_naive_repeats_last_value():
    pred = Naive().fit(to_panel(Y6)).predict(3)
    assert np.allclose(pred["yhat"], [6.0, 6.0, 6.0])


def test_naive_fitted_is_shifted_y():
    fitted = Naive().fit(to_panel(Y6)).fitted_values()["fitted"].to_numpy()
    assert np.isnan(fitted[0])
    assert np.allclose(fitted[1:], Y6[:-1])


def test_drift_hand_computed():
    pred = Drift().fit(to_panel(Y6)).predict(2)
    assert np.allclose(pred["yhat"], [7.0, 8.0])


def test_drift_fitted_hand_computed():
    # fitted[t] = y[t-1] + (y[t-1] - y[0]) / (t - 1) for t >= 2
    fitted = Drift().fit(to_panel(Y6)).fitted_values()["fitted"].to_numpy()
    assert np.isnan(fitted[:2]).all()
    assert np.allclose(fitted[2:], [3.0, 4.0, 5.0, 6.0])


def test_seasonal_naive_hand_computed():
    pred = SeasonalNaive(season_length=3).fit(to_panel(Y6)).predict(4)
    assert np.allclose(pred["yhat"], [4.0, 5.0, 6.0, 4.0])


def test_window_average_hand_computed():
    pred = WindowAverage(window=3).fit(to_panel(Y6)).predict(2)
    assert np.allclose(pred["yhat"], [5.0, 5.0])


def test_window_average_fitted_expanding_then_trailing():
    fitted = WindowAverage(window=2).fit(to_panel([1.0, 2.0, 3.0, 4.0]))
    fitted = fitted.fitted_values()["fitted"].to_numpy()
    assert np.isnan(fitted[0])
    assert np.allclose(fitted[1:], [1.0, 1.5, 2.5])


# -- guards -------------------------------------------------------------------


def test_drift_requires_two_observations():
    with pytest.raises(ForecastOSError):
        Drift().fit(to_panel([5.0]))


def test_seasonal_naive_short_series_raises():
    with pytest.raises(ForecastOSError):
        SeasonalNaive(season_length=7).fit(to_panel([1.0, 2.0, 3.0]))


@pytest.mark.parametrize("cls", [Naive, SeasonalNaive, Drift, WindowAverage])
def test_predict_before_fit_raises(cls):
    with pytest.raises(NotFittedError):
        cls().predict(3)


# -- intervals ----------------------------------------------------------------


def _noisy_series(n=30, seed=0):
    rng = np.random.default_rng(seed)
    return to_panel(10.0 + rng.standard_normal(n))


def test_naive_interval_width_grows_with_h():
    pred = Naive().fit(_noisy_series()).predict(5, level=[80])
    width = (pred["hi-80"] - pred["lo-80"]).to_numpy()
    assert (np.diff(width) > 0).all()


def test_drift_interval_width_grows_with_h():
    pred = Drift().fit(_noisy_series()).predict(5, level=[80])
    width = (pred["hi-80"] - pred["lo-80"]).to_numpy()
    assert (np.diff(width) > 0).all()


def test_seasonal_naive_sigma_steps_by_season():
    pred = SeasonalNaive(season_length=3).fit(_noisy_series(n=24)).predict(6, level=[80])
    width = (pred["hi-80"] - pred["lo-80"]).to_numpy()
    # constant within the first season, then wider in the second season
    assert np.allclose(width[:3], width[0])
    assert (width[3:] > width[:3]).all()


# -- panel behavior + registry -----------------------------------------------


@pytest.mark.parametrize("cls", [Naive, SeasonalNaive, Drift, WindowAverage])
def test_defaults_on_multiseries_panel(cls, panel):
    pred = cls().fit(panel).predict(8, level=[80])
    assert (pred.groupby("unique_id").size() == 8).all()
    assert np.isfinite(pred[["yhat", "lo-80", "hi-80"]]).all().all()
    assert (pred["lo-80"] <= pred["hi-80"]).all()


@pytest.mark.parametrize(
    "name, cls",
    [
        ("naive", Naive),
        ("seasonal_naive", SeasonalNaive),
        ("drift", Drift),
        ("window_average", WindowAverage),
    ],
)
def test_registered_as_baseline(name, cls):
    spec = _REGISTRY[name]
    assert spec.cls is cls
    assert spec.family == "baseline"
