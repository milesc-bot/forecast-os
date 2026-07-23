"""Tests for exponential smoothing models: SES, Holt, HoltWinters, AutoETS."""

import numpy as np
import pandas as pd
import pytest

from forecast_os.core.exceptions import ForecastOSError, NotFittedError
from forecast_os.core.registry import _REGISTRY
from forecast_os.core.types import to_panel
from forecast_os.datasets.synthetic import generate_series
from forecast_os.models.baselines import Naive
from forecast_os.models.ets import SES, AutoETS, Holt, HoltWinters


def _split(df, frac=0.8):
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


# -- SES ----------------------------------------------------------------------


def test_ses_alpha_one_equals_naive(panel):
    ses = SES(alpha=1.0).fit(panel).predict(5)
    naive = Naive().fit(panel).predict(5)
    assert np.allclose(ses["yhat"], naive["yhat"])


def test_ses_forecast_is_flat(panel):
    pred = SES().fit(panel).predict(6)
    for _, g in pred.groupby("unique_id"):
        assert np.allclose(g["yhat"], g["yhat"].iloc[0])


def test_ses_optimized_on_near_constant_series():
    rng = np.random.default_rng(1)
    df = to_panel(5.0 + 0.01 * rng.standard_normal(60))
    pred = SES().fit(df).predict(4)
    assert np.allclose(pred["yhat"], 5.0, atol=0.1)


def test_ses_interval_widths_nondecreasing():
    rng = np.random.default_rng(2)
    df = to_panel(10.0 + rng.standard_normal(50))
    pred = SES().fit(df).predict(10, level=[80])
    width = (pred["hi-80"] - pred["lo-80"]).to_numpy()
    assert (np.diff(width) >= -1e-12).all()


def test_ses_optimized_alpha_lives_in_state_not_self(flat_panel):
    model = SES().fit(flat_panel)
    assert model.alpha is None  # fit must stay per-series
    for state in model._series_state.values():
        assert 0.01 <= state["alpha_"] <= 0.99


# -- Holt ---------------------------------------------------------------------


def test_holt_recovers_clean_linear_trend():
    t = np.arange(50, dtype=float)
    df = to_panel(3.0 + 2.0 * t)
    pred = Holt().fit(df).predict(10)
    expected = 3.0 + 2.0 * np.arange(50, 60)
    assert np.allclose(pred["yhat"], expected, rtol=0.05)


def test_holt_min_train_size():
    with pytest.raises(ForecastOSError):
        Holt().fit(to_panel([1.0, 2.0]))


def test_holt_no_runaway_trend_on_seasonal_data():
    # Regression: with beta searched over [0.01, 0.99] the SSE optimizer hit
    # the corner beta ~= 1, making the trend equal the last first-difference
    # (a seasonal swing); h=14 forecasts collapsed to ~19-83 vs truth ~146-190.
    df = generate_series(n_series=5, length=300, seasonality=7, trend=0.3, seed=42)
    model = Holt().fit(df)
    pred = model.predict(14)
    last = df.groupby("unique_id")["y"].last()
    for uid, g in pred.groupby("unique_id"):
        assert np.abs(g["yhat"].to_numpy() - last[uid]).max() < 30.0
    for state in model._series_state.values():
        assert 0.01 <= state["beta_"] <= 0.3  # optimized beta stays capped


def test_holt_explicit_beta_is_unrestricted():
    # Only the *optimized* beta search is capped at 0.3.
    t = np.arange(30, dtype=float)
    model = Holt(alpha=0.5, beta=0.8).fit(to_panel(1.0 + 2.0 * t))
    for state in model._series_state.values():
        assert state["beta_"] == 0.8


def test_holt_interval_width_strictly_increases(trend_panel):
    pred = Holt().fit(trend_panel).predict(12, level=[95])
    for _, g in pred.groupby("unique_id"):
        width = (g["hi-95"] - g["lo-95"]).to_numpy()
        assert (np.diff(width) > 0).all()


def test_holt_90_coverage_on_local_trend_data():
    # Empirical sanity: 90% intervals on simulated local-linear-trend data
    # should cover roughly 90% of future observations (was ~69% with the old
    # constant-width fallback at 95%).
    rng = np.random.default_rng(0)
    h, hits, total = 10, 0, 0
    for _ in range(30):
        n = 120
        level, trend = 100.0, 0.5
        y = np.empty(n + h)
        for t in range(n + h):
            level += trend + 0.5 * rng.standard_normal()
            trend += 0.05 * rng.standard_normal()
            y[t] = level + rng.standard_normal()
        pred = Holt().fit(to_panel(y[:n])).predict(h, level=[90])
        inside = (y[n:] >= pred["lo-90"].to_numpy()) & (y[n:] <= pred["hi-90"].to_numpy())
        hits += int(inside.sum())
        total += h
    assert 0.75 <= hits / total <= 0.99


# -- HoltWinters --------------------------------------------------------------


@pytest.mark.parametrize("seasonal", ["add", "mul"])
def test_holt_winters_beats_naive_on_seasonal_panel(seasonal, panel):
    train, test = _split(panel, frac=0.8)
    hw_mae = _holdout_mae(HoltWinters(season_length=7, seasonal=seasonal), train, test)
    naive_mae = _holdout_mae(Naive(), train, test)
    assert hw_mae < 0.8 * naive_mae  # at least 20% better


def test_holt_winters_mul_raises_on_nonpositive_data():
    y = np.sin(np.arange(40)) * 5.0 - 1.0  # crosses zero
    with pytest.raises(ForecastOSError):
        HoltWinters(season_length=7, seasonal="mul").fit(to_panel(y))


def test_holt_winters_invalid_seasonal_mode():
    with pytest.raises(ValueError):
        HoltWinters(seasonal="bogus")


def test_holt_winters_min_train_size_is_two_seasons():
    rng = np.random.default_rng(3)
    with pytest.raises(ForecastOSError):
        HoltWinters(season_length=7).fit(to_panel(rng.standard_normal(13)))


def test_holt_winters_add_interval_width_strictly_increases(panel):
    pred = HoltWinters(season_length=7, seasonal="add").fit(panel).predict(15, level=[95])
    for _, g in pred.groupby("unique_id"):
        width = (g["hi-95"] - g["lo-95"]).to_numpy()
        assert (np.diff(width) > 0).all()


def test_holt_winters_mul_keeps_constant_sigma_fallback(panel):
    # No closed-form sigma for the multiplicative state space.
    model = HoltWinters(season_length=7, seasonal="mul").fit(panel)
    for state in model._series_state.values():
        sigma = model._predict_sigma(state, 10)
        assert np.allclose(sigma, state["_sigma"])


# -- AutoETS ------------------------------------------------------------------


def test_autoets_picks_seasonal_candidate_on_seasonal_data():
    df = generate_series(
        n_series=2, length=140, freq="D", trend=0.2, seasonality=7, season_amp=25.0,
        noise=0.5, seed=101,
    )
    model = AutoETS(season_length=7).fit(df)
    for state in model._series_state.values():
        assert state["winner_name"] == "HoltWinters"


def test_autoets_runs_with_default_params(panel):
    pred = AutoETS().fit(panel).predict(8, level=[80])
    assert (pred.groupby("unique_id").size() == 8).all()
    assert np.isfinite(pred[["yhat", "lo-80", "hi-80"]]).all().all()
    assert (pred["lo-80"] <= pred["hi-80"]).all()


def test_autoets_prefers_trend_model_on_trending_data(trend_panel):
    model = AutoETS().fit(trend_panel)
    for state in model._series_state.values():
        assert state["winner_name"] == "Holt"


def test_autoets_sigma_delegation_uses_winner_formula(trend_panel):
    # AutoETS delegates _predict_sigma to the winner, so a Holt winner must
    # produce horizon-growing (not constant) interval widths.
    model = AutoETS().fit(trend_panel)
    assert all(s["winner_name"] == "Holt" for s in model._series_state.values())
    pred = model.predict(10, level=[90])
    for _, g in pred.groupby("unique_id"):
        width = (g["hi-90"] - g["lo-90"]).to_numpy()
        assert (np.diff(width) > 0).all()


# -- shared -------------------------------------------------------------------


@pytest.mark.parametrize("cls", [SES, Holt, HoltWinters, AutoETS])
def test_predict_before_fit_raises(cls):
    with pytest.raises(NotFittedError):
        cls().predict(3)


@pytest.mark.parametrize(
    "name, cls",
    [("ses", SES), ("holt", Holt), ("holt_winters", HoltWinters), ("auto_ets", AutoETS)],
)
def test_registered_as_statistical(name, cls):
    spec = _REGISTRY[name]
    assert spec.cls is cls
    assert spec.family == "statistical"
