"""Tests for the classical Theta method."""

import numpy as np
import pandas as pd
import pytest

from forecast_os.core.exceptions import NotFittedError
from forecast_os.core.registry import _REGISTRY
from forecast_os.core.types import to_panel
from forecast_os.models.baselines import Naive
from forecast_os.models.ets import _fit_ses
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
    # sigma_k = sigma * sqrt(1 + (k-1) * alpha^2): the recombined forecast is
    # the SES level of the original series plus a deterministic drift, so it
    # accumulates variance exactly like SES.
    pred = Theta().fit(trend_panel).predict(12, level=[90])
    for _, g in pred.groupby("unique_id"):
        width = (g["hi-90"] - g["lo-90"]).to_numpy()
        assert (np.diff(width) > 0).all()


def test_theta_sigma_growth_coefficient_is_alpha_not_alpha_over_theta(trend_panel):
    """The combination weight must not scale the horizon-growth coefficient.

    This test previously pinned ``(alpha * 1/theta)^2``, which encoded a
    derivation error: because SES is linear, the level of the theta line splits
    as ell^q = theta*ell^x + (1-theta)*SES(line0), and substituting it into
    ``w*ell^q + (1-w)*line0_{n+k}`` with w = 1/theta cancels the theta weights
    on the stochastic term, leaving the SES level of the ORIGINAL series plus a
    deterministic drift (see ``test_theta_forecast_equals_ses_level_plus_drift``).
    ``_sigma`` is already the residual std of the recombined fit in original
    units, so multiplying alpha by w double-counted the weight and made every
    multi-step interval too narrow (nominal-80 coverage 0.53/0.50/0.45 at
    h=6/12/24 on a random walk, versus 0.77/0.76/0.77 with alpha unscaled).
    """
    model = Theta().fit(trend_panel)
    for state in model._series_state.values():
        sigma = model._predict_sigma(state, 8)
        k = np.arange(1, 9)
        expected = state["_sigma"] * np.sqrt(1 + (k - 1) * state["alpha_"] ** 2)
        assert np.allclose(sigma, expected)


def test_theta_forecast_equals_ses_level_plus_drift():
    """Evidence for the sigma derivation: the theta weights cancel exactly.

    ``w * SES(theta line) + (1 - w) * line0`` reduces to
    ``SES(x) + (1 - w) * (line0_{n+k} - SES(line0))`` at the same alpha, so the
    stochastic part of the forecast is a plain SES level and its psi weights
    are ``(1, alpha, alpha, ...)``.
    """
    rng = np.random.default_rng(7)
    y = np.cumsum(rng.normal(size=100))
    model = Theta().fit(to_panel(y))
    state = next(iter(model._series_state.values()))
    h, n = 24, len(y)
    t = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(t, y, 1)
    line0 = intercept + slope * t
    alpha = state["alpha_"]
    _, level_x, _ = _fit_ses(y, alpha)
    _, level_line, _ = _fit_ses(line0, alpha)
    k = np.arange(1, h + 1)
    w = 1.0 / model.theta
    direct = level_x + (1 - w) * (intercept + slope * (n - 1 + k) - level_line)
    assert np.allclose(model._predict_series(state, h), direct, atol=1e-10)


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
