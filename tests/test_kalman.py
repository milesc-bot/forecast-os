"""Tests for models/kalman.py: MLE Kalman filter forecaster with native intervals."""

import numpy as np
import pytest

from forecast_os.core.exceptions import ForecastOSError, NotFittedError
from forecast_os.core.registry import get_model, list_models
from forecast_os.core.types import to_panel
from forecast_os.datasets.synthetic import generate_series
from forecast_os.models.kalman import KalmanForecaster

H = 8


def _contract_panel():
    return generate_series(
        n_series=3, length=80, freq="D", trend=0.3, seasonality=7, season_amp=5.0,
        noise=0.8, seed=21,
    )


def _naive_mae(train, test):
    last = train.groupby("unique_id")["y"].last()
    return float((test["y"] - test["unique_id"].map(last)).abs().mean())


class TestKalmanForecaster:
    def test_registered_as_statistical(self):
        assert isinstance(get_model("kalman"), KalmanForecaster)
        listed = list_models(family="statistical")
        assert "kalman" in set(listed["name"])

    def test_invalid_model_raises(self):
        with pytest.raises(ForecastOSError):
            KalmanForecaster(model="structural")

    def test_params_stored_and_clone(self):
        model = KalmanForecaster(model="local_linear")
        assert model.get_params() == {"model": "local_linear"}
        clone = model.clone()
        assert type(clone) is KalmanForecaster
        assert clone.get_params() == model.get_params()

    def test_predict_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            KalmanForecaster().predict(3)

    def test_local_level_on_noisy_constant(self):
        rng = np.random.default_rng(1)
        y = 10.0 + 0.5 * rng.standard_normal(200)
        model = KalmanForecaster(model="local_level").fit(to_panel(y))
        yhat = model.predict(10)["yhat"].to_numpy()
        assert np.allclose(yhat, yhat[0]), "local level forecast must be flat"
        assert abs(yhat[0] - 10.0) < 0.5

    def test_local_linear_tracks_trend(self):
        rng = np.random.default_rng(2)
        n = 120
        y = 5.0 + 2.0 * np.arange(n, dtype=float) + 0.2 * rng.standard_normal(n)
        model = KalmanForecaster(model="local_linear").fit(to_panel(y))
        yhat = model.predict(10)["yhat"].to_numpy()
        truth = 5.0 + 2.0 * np.arange(n, n + 10, dtype=float)
        assert np.allclose(yhat, truth, rtol=0.05)

    def test_interval_width_grows_with_h(self):
        rng = np.random.default_rng(3)
        y = np.cumsum(rng.standard_normal(300))
        model = KalmanForecaster(model="local_level").fit(to_panel(y))
        pred = model.predict(20, level=[80])
        width = (pred["hi-80"] - pred["lo-80"]).to_numpy()
        assert np.all(np.diff(width) > 0), "interval width must grow with horizon"
        assert width[-1] > 1.2 * width[0]

    def test_local_linear_beats_naive_on_trend(self, trend_panel):
        train = trend_panel.groupby("unique_id").head(100)
        test = trend_panel.groupby("unique_id").tail(20)
        pred = KalmanForecaster(model="local_linear").fit(train).predict(20)
        merged = test.merge(pred, on=["unique_id", "ds"], validate="one_to_one")
        assert len(merged) == len(test)
        mae = float((merged["y"] - merged["yhat"]).abs().mean())
        assert mae < _naive_mae(train, test)

    def test_fitted_values_are_one_step_predictions(self):
        rng = np.random.default_rng(6)
        y = 10.0 + 0.5 * rng.standard_normal(100)
        model = KalmanForecaster(model="local_level").fit(to_panel(y))
        fitted = model.fitted_values()["fitted"].to_numpy()
        assert np.isnan(fitted[0])
        assert np.isfinite(fitted[1:]).all()
        # one-step predictions of a noisy constant should hug the level
        assert abs(np.nanmean(fitted) - 10.0) < 0.5

    def test_default_on_contract_panel(self):
        df = _contract_panel()
        pred = KalmanForecaster().fit(df).predict(H, level=[80])
        assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
        assert (pred.groupby("unique_id").size() == H).all()
        for col in ("yhat", "lo-80", "hi-80"):
            assert np.isfinite(pred[col]).all()
        assert (pred["lo-80"] <= pred["yhat"]).all()
        assert (pred["yhat"] <= pred["hi-80"]).all()
