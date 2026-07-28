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
    # Regression: this test used to check finiteness only, which passed while
    # every horizon reported the identical one-step residual sigma.
    for _, g in pred.groupby(ID_COL):
        width = (g["hi-80"] - g["lo-80"]).to_numpy()
        assert (np.diff(width) >= 0).all()
        assert width[-1] > width[0]


def test_interval_width_grows_with_horizon_on_persistent_series():
    """RidgeLag reported the one-step sigma at every horizon.

    ``_predict_series`` feeds its own predictions back into the lag window, so
    a one-step innovation propagates: the h-step variance is
    sigma^2 * sum_{j<h} psi_j^2 with psi from the fitted lag weights. With no
    ``_predict_sigma`` the model inherited the flat ``np.full(h, sigma)``
    fallback, making the h=24 band bit-identical to the h=1 band on an AR(1)
    with phi=0.8 (nominal-80 coverage measured at 0.60 for h>=6). The correct
    behaviour is strictly widening intervals whenever the fitted AR weights
    are non-trivial.
    """
    y = _ar1(400, phi=0.8, seed=11)
    model = RidgeLag(lags=5).fit(to_panel(y))
    pred = model.predict(24, level=[80])
    width = (pred["hi-80"] - pred["lo-80"]).to_numpy()
    assert (np.diff(width) > 0).all()
    assert width[23] > 1.5 * width[0]


def test_interval_width_matches_psi_weight_recursion():
    """Pin the psi-weight construction (same shape as ARIMA's ``_psi_sigma``).

    Only the ``lags`` leading weights enter: Fourier and exogenous columns are
    deterministic given the future clock, so they carry no innovation.
    """
    lags = 4
    y = _ar1(300, phi=0.7, seed=3)
    model = RidgeLag(lags=lags, season_length=7, fourier_k=2).fit(to_panel(y))
    state = next(iter(model._series_state.values()))
    h = 12
    phi = state["w"][:lags] * state["y_std"] / state["x_std"][:lags]
    psi = np.zeros(h)
    psi[0] = 1.0
    for j in range(1, h):
        psi[j] = sum(phi[i - 1] * psi[j - i] for i in range(1, min(j, lags) + 1))
    expected = state["_sigma"] * np.sqrt(np.cumsum(psi**2))
    assert np.allclose(model._predict_sigma(state, h), expected)
    assert np.isclose(model._predict_sigma(state, h)[0], state["_sigma"])


def test_flat_series_keeps_flat_intervals():
    """Zero fitted lag weights must leave the constant-sigma behaviour intact.

    The psi recursion degenerates to psi = (1, 0, 0, ...) when the model has no
    autoregressive signal, so the fix must not widen intervals on data whose
    dynamics do not imply widening.
    """
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)  # iid: fitted lag weights shrink to ~0
    model = RidgeLag(lags=5, alpha=10.0).fit(to_panel(y))
    pred = model.predict(20, level=[80])
    width = (pred["hi-80"] - pred["lo-80"]).to_numpy()
    assert np.allclose(width, width[0], rtol=0.05)


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
