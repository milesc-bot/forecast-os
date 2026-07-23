"""Tests for the classical Theta method."""

import numpy as np
import pandas as pd
import pytest

from forecast_os.core.exceptions import NotFittedError
from forecast_os.core.registry import _REGISTRY
from forecast_os.core.types import to_panel
from forecast_os.models.baselines import Naive
from forecast_os.models.theta import Theta


def _split(df, frac=0.75):
    train, test = [], []
    for _, g in df.groupby("unique_id"):
        n = int(len(g) * frac)
        train.append(g.iloc[:n])
        test.append(g.iloc[n:])
    return pd.concat(train, ignore_index=True), pd.concat(test, ignore_index=True)


def _holdout_mae(model, train, test):
    h = int(test.groupby("unique_id").size().iloc[0])
    pred = model.fit(train).predict(h)
    merged = test.merge(pred, on=["unique_id", "ds"])
    assert len(merged) == len(test)
    return float(np.abs(merged["y"] - merged["yhat"]).mean())


def test_theta_beats_naive_on_trending_data(trend_panel):
    train, test = _split(trend_panel)
    assert _holdout_mae(Theta(), train, test) < _holdout_mae(Naive(), train, test)


def test_theta_with_seasonality_beats_naive_on_seasonal_panel(panel):
    train, test = _split(panel, frac=0.8)
    theta_mae = _holdout_mae(Theta(season_length=7), train, test)
    assert theta_mae < _holdout_mae(Naive(), train, test)


def test_theta_halves_the_trend_on_a_clean_line():
    # classical theta (theta=2) extrapolates half the OLS slope
    t = np.arange(60, dtype=float)
    pred = Theta().fit(to_panel(t)).predict(10)
    expected = 59.0 + 0.5 * np.arange(1, 11)
    assert np.allclose(pred["yhat"], expected, atol=1.0)


def test_theta_interval_width_strictly_increases(trend_panel):
    # sigma_k = sigma * sqrt(1 + (k-1) * (alpha * (1/theta))^2): the SES
    # component accumulates variance while the trend line is deterministic.
    pred = Theta().fit(trend_panel).predict(12, level=[90])
    for _, g in pred.groupby("unique_id"):
        width = (g["hi-90"] - g["lo-90"]).to_numpy()
        assert (np.diff(width) > 0).all()


def test_theta_sigma_scales_ses_growth_by_component_weight(trend_panel):
    model = Theta().fit(trend_panel)
    for state in model._series_state.values():
        sigma = model._predict_sigma(state, 8)
        k = np.arange(1, 9)
        w = 1.0 / model.theta
        expected = state["_sigma"] * np.sqrt(1 + (k - 1) * (state["alpha_"] * w) ** 2)
        assert np.allclose(sigma, expected)


def test_theta_default_params_full_contract(panel):
    pred = Theta().fit(panel).predict(8, level=[80])
    assert (pred.groupby("unique_id").size() == 8).all()
    assert np.isfinite(pred[["yhat", "lo-80", "hi-80"]]).all().all()
    assert (pred["lo-80"] <= pred["hi-80"]).all()


def test_theta_invalid_theta_raises():
    with pytest.raises(ValueError):
        Theta(theta=0.5)


def test_theta_predict_before_fit_raises():
    with pytest.raises(NotFittedError):
        Theta().predict(3)


def test_theta_registered_as_statistical():
    spec = _REGISTRY["theta"]
    assert spec.cls is Theta
    assert spec.family == "statistical"
