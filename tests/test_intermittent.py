"""Tests for intermittent-demand models: Croston and TSB."""

import numpy as np
import pandas as pd
import pytest

from forecast_os.core.exceptions import ForecastOSError, NotFittedError
from forecast_os.core.registry import _REGISTRY
from forecast_os.core.types import to_panel
from forecast_os.models.baselines import Naive
from forecast_os.models.intermittent import TSB, Croston

# Hand-traced Croston recursion, alpha = 0.1:
#   demands at t=1 (y=2), t=4 (y=3, gap 3), t=6 (y=4, gap 2)
#   sizes:     2 -> 0.1*3 + 0.9*2 = 2.1 -> 0.1*4 + 0.9*2.1 = 2.29
#   intervals: 2 -> 0.1*3 + 0.9*2 = 2.1 -> 0.1*2 + 0.9*2.1 = 2.09
CROSTON_Y = [0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0]
CROSTON_FORECAST = 2.29 / 2.09


def test_croston_hand_computed_forecast():
    pred = Croston(alpha=0.1).fit(to_panel(CROSTON_Y)).predict(3)
    np.testing.assert_allclose(pred["yhat"], CROSTON_FORECAST)


def test_croston_fitted_values_hand_computed():
    model = Croston(alpha=0.1).fit(to_panel(CROSTON_Y))
    fitted = model.fitted_values()["fitted"].to_numpy()
    # warm-up NaN through the first demand; then z/p held between demands
    assert np.isnan(fitted[:2]).all()
    np.testing.assert_allclose(fitted[2:7], 1.0)  # 2/2 then 2.1/2.1
    np.testing.assert_allclose(fitted[7], CROSTON_FORECAST)


def test_croston_forecast_is_flat():
    pred = Croston(alpha=0.1).fit(to_panel(CROSTON_Y)).predict(6)
    assert np.allclose(pred["yhat"], pred["yhat"].iloc[0])


# Hand-traced TSB recursion, alpha = 0.2 (sizes), beta = 0.1 (probability):
#   init p = 2/6, z = mean(3, 5) = 4
#   p: 0.3 -> 0.37 -> 0.333 -> 0.2997 -> 0.36973 -> 0.332757
#   z: 4 -> 3.8 (t=1) -> 4.04 (t=4)
TSB_Y = [0.0, 3.0, 0.0, 0.0, 5.0, 0.0]
TSB_FORECAST = 0.332757 * 4.04


def test_tsb_hand_computed_forecast():
    pred = TSB(alpha=0.2, beta=0.1).fit(to_panel(TSB_Y)).predict(2)
    np.testing.assert_allclose(pred["yhat"], TSB_FORECAST)


def test_tsb_fitted_values_hand_computed():
    model = TSB(alpha=0.2, beta=0.1).fit(to_panel(TSB_Y))
    fitted = model.fitted_values()["fitted"].to_numpy()
    np.testing.assert_allclose(fitted[0], (2.0 / 6.0) * 4.0)
    np.testing.assert_allclose(fitted[1], 0.3 * 4.0)
    np.testing.assert_allclose(fitted[4], 0.2997 * 3.8)
    np.testing.assert_allclose(fitted[5], 0.36973 * 4.04)


def test_tsb_forecast_is_flat():
    pred = TSB().fit(to_panel(TSB_Y)).predict(5)
    assert np.allclose(pred["yhat"], pred["yhat"].iloc[0])


# -- guards -------------------------------------------------------------------


@pytest.mark.parametrize("model_cls", [Croston, TSB])
def test_all_zero_series_forecasts_zero(model_cls):
    pred = model_cls().fit(to_panel([0.0] * 10)).predict(4, level=[80])
    np.testing.assert_allclose(pred["yhat"], 0.0)
    assert np.isfinite(pred[["lo-80", "hi-80"]]).all().all()


@pytest.mark.parametrize("model_cls", [Croston, TSB])
def test_negative_demand_raises(model_cls):
    with pytest.raises(ForecastOSError, match="nonnegative"):
        model_cls().fit(to_panel([0.0, 2.0, -1.0, 3.0, 0.0]))


@pytest.mark.parametrize("model_cls", [Croston, TSB])
def test_min_train_size_enforced(model_cls):
    with pytest.raises(ForecastOSError, match="at least 4"):
        model_cls().fit(to_panel([0.0, 1.0, 0.0]))


@pytest.mark.parametrize("model_cls", [Croston, TSB])
def test_predict_before_fit_raises(model_cls):
    with pytest.raises(NotFittedError):
        model_cls().predict(3)


def test_invalid_smoothing_params_raise():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            Croston(alpha=bad)
        with pytest.raises(ValueError):
            TSB(alpha=bad)
        with pytest.raises(ValueError):
            TSB(beta=bad)


# -- registry & contract conventions ------------------------------------------


def test_registered_names_and_family():
    assert _REGISTRY["croston"].cls is Croston
    assert _REGISTRY["tsb"].cls is TSB
    assert _REGISTRY["croston"].family == "statistical"
    assert _REGISTRY["tsb"].family == "statistical"


def test_get_params_and_clone():
    m = Croston(alpha=0.25)
    assert m.get_params() == {"alpha": 0.25}
    assert type(m.clone()) is Croston and m.clone().alpha == 0.25
    t = TSB(alpha=0.3, beta=0.05)
    assert t.get_params() == {"alpha": 0.3, "beta": 0.05}
    assert t.clone().get_params() == t.get_params()


def test_multi_series_panel_and_future_ds():
    ds = pd.date_range("2025-01-06", periods=8, freq="W-MON")
    df = pd.concat(
        [
            pd.DataFrame({"unique_id": "lumpy", "ds": ds, "y": CROSTON_Y}),
            pd.DataFrame({"unique_id": "dead", "ds": ds, "y": 0.0}),
        ],
        ignore_index=True,
    )
    for model in (Croston(alpha=0.1), TSB(alpha=0.1, beta=0.1)):
        pred = model.fit(df).predict(3, level=[80])
        assert (pred.groupby("unique_id").size() == 3).all()
        assert (pred[pred["unique_id"] == "dead"]["yhat"] == 0.0).all()
        assert (pred[pred["unique_id"] == "lumpy"]["yhat"] > 0.0).all()
        assert (pred["ds"] > ds[-1]).all()
        assert (pred["lo-80"] <= pred["hi-80"]).all()


# -- accuracy on simulated lumpy demand ---------------------------------------


def _lumpy_split(n_series=4, length=120, n_train=100, p=0.35, lam=4.0, seed=101):
    """Zero-inflated Poisson-like demand; each series ends training on a spike
    (the case where naive carry-forward is worst and Croston/TSB shine)."""
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2025-01-01", periods=length, freq="D")
    frames = []
    for i in range(n_series):
        occur = rng.random(length) < p
        sizes = rng.poisson(lam, length).astype(float) + 1.0
        y = np.where(occur, sizes, 0.0)
        y[n_train - 1] = sizes[n_train - 1]
        frames.append(pd.DataFrame({"unique_id": f"sku-{i}", "ds": ds, "y": y}))
    df = pd.concat(frames, ignore_index=True)
    cutoff = ds[n_train - 1]
    return df[df["ds"] <= cutoff], df[df["ds"] > cutoff]


def _holdout_mae(model, train, test):
    h = int(test.groupby("unique_id").size().iloc[0])
    pred = model.fit(train).predict(h)
    merged = test.merge(pred, on=["unique_id", "ds"])
    assert len(merged) == len(test)
    return float(np.abs(merged["y"] - merged["yhat"]).mean())


@pytest.mark.parametrize(
    "model", [Croston(alpha=0.1), TSB(alpha=0.1, beta=0.1)], ids=["croston", "tsb"]
)
def test_beats_naive_on_lumpy_demand(model):
    train, test = _lumpy_split()
    assert _holdout_mae(model, train, test) < _holdout_mae(Naive(), train, test)
