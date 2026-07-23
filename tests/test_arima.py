"""Tests for models/arima.py: CSS-estimated ARIMA and AutoARIMA order selection."""

import numpy as np
import pytest

from forecast_os.core.exceptions import ForecastOSError, NotFittedError
from forecast_os.core.registry import get_model, list_models
from forecast_os.core.types import to_panel
from forecast_os.datasets.synthetic import generate_series
from forecast_os.models.arima import ARIMA, AutoARIMA

H = 8


def _ar1(n=500, phi=0.7, mu=0.0, sigma=1.0, seed=0):
    """Simulate a stationary AR(1) series around mean ``mu``."""
    rng = np.random.default_rng(seed)
    y = np.empty(n)
    y[0] = mu
    for t in range(1, n):
        y[t] = mu + phi * (y[t - 1] - mu) + sigma * rng.standard_normal()
    return y


def _contract_panel():
    return generate_series(
        n_series=3, length=80, freq="D", trend=0.3, seasonality=7, season_amp=5.0,
        noise=0.8, seed=21,
    )


def _naive_mae(train, test):
    last = train.groupby("unique_id")["y"].last()
    return float((test["y"] - test["unique_id"].map(last)).abs().mean())


def _model_mae(model, train, test, h):
    pred = model.fit(train).predict(h)
    merged = test.merge(pred, on=["unique_id", "ds"], validate="one_to_one")
    assert len(merged) == len(test)
    return float((merged["y"] - merged["yhat"]).abs().mean())


class TestARIMA:
    def test_registered_as_statistical(self):
        assert isinstance(get_model("arima"), ARIMA)
        listed = list_models(family="statistical")
        assert "arima" in set(listed["name"])

    def test_params_stored_and_clone(self):
        model = ARIMA(order=(2, 1, 1), include_mean=False)
        assert model.get_params() == {"order": (2, 1, 1), "include_mean": False}
        clone = model.clone()
        assert type(clone) is ARIMA
        assert clone.get_params() == model.get_params()

    def test_invalid_order_raises(self):
        with pytest.raises(ForecastOSError):
            ARIMA(order=(1, -1, 0))
        with pytest.raises(ForecastOSError):
            ARIMA(order=(1, 1))

    def test_min_train_size(self):
        # max(p, q) + d + 5 = 6 for (1, 0, 0); 5 observations must fail
        with pytest.raises(ForecastOSError):
            ARIMA(order=(1, 0, 0)).fit(to_panel(np.arange(5.0)))

    def test_predict_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            ARIMA().predict(3)

    def test_ar1_phi_recovery(self):
        y = _ar1(n=500, phi=0.7, seed=0)
        model = ARIMA(order=(1, 0, 0)).fit(to_panel(y))
        phi_hat = model._series_state["series-0"]["phi"][0]
        assert abs(phi_hat - 0.7) <= 0.1

    def test_010_is_random_walk_flat_forecast(self):
        # ARIMA(0,1,0): no c when d > 0, so the forecast is flat at the last value
        rng = np.random.default_rng(4)
        y = 100.0 + 1.0 * np.arange(120.0) + 0.5 * rng.standard_normal(120)
        model = ARIMA(order=(0, 1, 0)).fit(to_panel(y))
        pred = model.predict(6)
        assert np.allclose(pred["yhat"], y[-1])

    def test_include_mean_silently_ignored_when_differenced(self):
        # Documented semantics (mirrors R's arima include.mean): the constant
        # is estimated only when d == 0; with d > 0 it is silently dropped.
        rng = np.random.default_rng(8)
        y = 50.0 + np.cumsum(rng.standard_normal(120))
        model = ARIMA(order=(1, 1, 0), include_mean=True).fit(to_panel(y))
        assert model._series_state["series-0"]["c"] == 0.0

    def test_010_sigma_grows_like_sqrt_k(self):
        # psi weights of (0,1,0) cumsum to all-ones: sigma_k = resid_std * sqrt(k)
        rng = np.random.default_rng(4)
        y = np.cumsum(rng.standard_normal(150))
        model = ARIMA(order=(0, 1, 0)).fit(to_panel(y))
        pred = model.predict(10, level=[80])
        width = (pred["hi-80"] - pred["lo-80"]).to_numpy()
        ratio = width / width[0]
        assert np.allclose(ratio, np.sqrt(np.arange(1, 11)), rtol=1e-8)

    def test_forecast_converges_to_series_mean(self):
        y = _ar1(n=500, phi=0.7, mu=5.0, seed=3)
        model = ARIMA(order=(1, 0, 0)).fit(to_panel(y))
        state = model._series_state["series-0"]
        mu_hat = state["c"] / (1 - state["phi"][0])
        yhat = model.predict(80)["yhat"].to_numpy()
        gaps = np.abs(yhat - mu_hat)
        assert np.all(np.diff(gaps) <= 1e-9), "distance to the mean must shrink monotonically"
        assert gaps[-1] < 1e-2
        assert abs(mu_hat - y.mean()) < 0.5

    def test_fitted_values_warmup(self):
        y = _ar1(n=200, phi=0.5, seed=6)
        model = ARIMA(order=(2, 1, 1)).fit(to_panel(y))
        fv = model.fitted_values()
        fitted = fv["fitted"].to_numpy()
        # warm-up: d + max(p, q) = 1 + 2 = 3 NaNs, everything else finite
        assert np.isnan(fitted[:3]).all()
        assert np.isfinite(fitted[3:]).all()

    def test_beats_naive_on_trend(self, trend_panel):
        train = trend_panel.groupby("unique_id").head(100)
        test = trend_panel.groupby("unique_id").tail(20)
        mae = _model_mae(ARIMA(), train, test, 20)
        assert mae < _naive_mae(train, test)

    def test_default_on_contract_panel(self):
        df = _contract_panel()
        pred = ARIMA().fit(df).predict(H, level=[80])
        assert (pred.groupby("unique_id").size() == H).all()
        for col in ("yhat", "lo-80", "hi-80"):
            assert np.isfinite(pred[col]).all()
        assert (pred["lo-80"] <= pred["yhat"]).all()
        assert (pred["yhat"] <= pred["hi-80"]).all()
        for _, g in pred.groupby("unique_id"):
            width = (g["hi-80"] - g["lo-80"]).to_numpy()
            assert np.all(np.diff(width) >= -1e-9), "interval width must not shrink with h"


class TestAutoARIMA:
    def test_registered_as_statistical(self):
        assert isinstance(get_model("auto_arima"), AutoARIMA)
        listed = list_models(family="statistical")
        assert "auto_arima" in set(listed["name"])

    def test_params_stored_and_clone(self):
        model = AutoARIMA(max_p=2, max_d=1, max_q=2)
        assert model.get_params() == {"max_p": 2, "max_d": 1, "max_q": 2}
        assert model.clone().get_params() == model.get_params()

    def test_predict_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            AutoARIMA().predict(3)

    def test_min_train_size(self):
        # max(max_p, max_q) + max_d + 5 = 10 by default; 9 observations must fail
        with pytest.raises(ForecastOSError):
            AutoARIMA().fit(to_panel(np.arange(9.0)))

    def test_ar1_picks_stationary_ar(self):
        # phi=0.4 keeps std(diff(y)) * 1.05 above std(y), so the heuristic keeps d=0
        y = _ar1(n=400, phi=0.4, seed=2)
        model = AutoARIMA().fit(to_panel(y))
        p, d, _q = model._series_state["series-0"]["order_"]
        assert d == 0
        assert p >= 1

    def test_random_walk_picks_differencing(self):
        rng = np.random.default_rng(5)
        y = np.cumsum(rng.standard_normal(400))
        model = AutoARIMA().fit(to_panel(y))
        _p, d, _q = model._series_state["series-0"]["order_"]
        assert d >= 1

    def test_default_on_contract_panel(self):
        df = _contract_panel()
        pred = AutoARIMA().fit(df).predict(H, level=[80])
        assert (pred.groupby("unique_id").size() == H).all()
        for col in ("yhat", "lo-80", "hi-80"):
            assert np.isfinite(pred[col]).all()
        assert (pred["lo-80"] <= pred["hi-80"]).all()
