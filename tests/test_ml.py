"""Tests for RidgeLag (lag-feature ridge autoregressor)."""

import numpy as np
import pytest

from forecast_os.core.exceptions import ForecastOSError, NotFittedError
from forecast_os.core.types import ID_COL, TARGET_COL, to_panel
from forecast_os.datasets.synthetic import generate_series
from forecast_os.evaluation.metrics import mae, rmse
from forecast_os.models.ml import RidgeLag


def _ar1(n: int, phi: float = 0.8, sigma: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + sigma * rng.standard_normal()
    return y


def test_beats_naive_rmse_on_ar1_holdout():
    h = 30
    model_rmse, naive_rmse = [], []
    for seed in range(5):
        y = _ar1(400, phi=0.8, seed=seed)
        train, test = y[:-h], y[-h:]
        model = RidgeLag(lags=5).fit(to_panel(train))
        pred = model.predict(h)
        model_rmse.append(rmse(test, pred["yhat"]))
        naive_rmse.append(rmse(test, np.full(h, train[-1])))
    assert np.mean(model_rmse) < np.mean(naive_rmse)


def test_beats_naive_mae_on_seasonal_panel_by_20pct():
    df = generate_series(
        n_series=3, length=140, freq="D", trend=0.5, seasonality=7, season_amp=10.0,
        noise=1.0, seed=5,
    )
    h = 28
    n = df.groupby(ID_COL)[TARGET_COL].transform("size").to_numpy()
    pos = df.groupby(ID_COL).cumcount().to_numpy()
    train, test = df[pos < n - h], df[pos >= n - h]

    model = RidgeLag(lags=14, season_length=7).fit(train)
    pred = model.predict(h)

    y_true = test[TARGET_COL].to_numpy()
    yhat = pred["yhat"].to_numpy()
    naive = test[ID_COL].map(train.groupby(ID_COL)[TARGET_COL].last()).to_numpy()
    assert mae(y_true, yhat) < 0.8 * mae(y_true, naive)


def test_recursive_forecast_finite_h30(panel):
    model = RidgeLag().fit(panel)
    pred = model.predict(30, level=[80])
    assert len(pred) == 3 * 30
    for col in ("yhat", "lo-80", "hi-80"):
        assert np.isfinite(pred[col]).all()
    assert (pred["lo-80"] <= pred["hi-80"]).all()


def test_fitted_values_have_lag_warmup(panel):
    model = RidgeLag(lags=14).fit(panel)
    fv = model.fitted_values()
    first_uid = fv[ID_COL].iloc[0]
    g = fv[fv[ID_COL] == first_uid]
    assert g["fitted"].head(14).isna().all()
    assert np.isfinite(g["fitted"].iloc[14:]).all()


def test_predict_before_fit_raises():
    with pytest.raises(NotFittedError):
        RidgeLag().predict(3)


def test_too_short_series_raises():
    with pytest.raises(ForecastOSError):
        RidgeLag(lags=5).fit(to_panel(np.arange(14.0)))


def test_get_params_clone_roundtrip():
    model = RidgeLag(lags=7, alpha=0.5, season_length=7, fourier_k=2)
    clone = model.clone()
    assert clone is not model
    assert clone.get_params() == model.get_params()
    assert clone.min_train_size == 17
