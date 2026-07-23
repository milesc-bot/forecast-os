"""Contract tests: every registered model honors the panel + forecaster contract.

Parametrized over the full registry (populated by importing ``forecast_os``),
so third-party or newly added models are automatically covered.
"""

import numpy as np
import pandas as pd
import pytest

import forecast_os  # noqa: F401  (imports register all built-in models)
from forecast_os.core.base import BaseForecaster
from forecast_os.core.registry import _REGISTRY, get_model
from forecast_os.core.types import ID_COL, TARGET_COL, TIME_COL
from forecast_os.datasets.synthetic import generate_series

H = 8


def _contract_panel():
    return generate_series(
        n_series=3, length=80, freq="D", trend=0.3, seasonality=7, season_amp=5.0,
        noise=0.8, seed=21,
    )


def _all_model_names():
    names = sorted(_REGISTRY)
    assert names, "registry is empty — forecast_os import did not register models"
    return names


@pytest.mark.parametrize("name", _all_model_names())
def test_registered_model_contract(name):
    model = get_model(name)
    assert isinstance(model, BaseForecaster)

    df = _contract_panel()
    fitted = model.fit(df)
    assert fitted is model, "fit() must return self"

    pred = model.predict(H, level=[80])
    assert list(pred.columns[:3]) == [ID_COL, TIME_COL, "yhat"]
    assert {"lo-80", "hi-80"} <= set(pred.columns)

    # h rows per series, all finite point forecasts
    counts = pred.groupby(ID_COL).size()
    assert set(counts.index) == set(df[ID_COL].unique())
    assert (counts == H).all()
    assert np.isfinite(pred["yhat"]).all(), f"{name} produced non-finite forecasts"
    assert (pred["lo-80"] <= pred["hi-80"] + 1e-9).all()

    # future ds strictly after the training data, strictly increasing
    last_train = df.groupby(ID_COL)[TIME_COL].max()
    for uid, g in pred.groupby(ID_COL):
        assert (g[TIME_COL] > last_train[uid]).all()
        assert g[TIME_COL].is_monotonic_increasing

    # clone() produces an equivalent unfitted instance
    clone = model.clone()
    assert type(clone) is type(model)
    assert clone.get_params() == model.get_params()
    pred2 = clone.fit(df).predict(H)
    pd.testing.assert_series_equal(pred["yhat"], pred2["yhat"], check_exact=False, atol=1e-6)


@pytest.mark.parametrize("name", _all_model_names())
def test_registered_model_predict_before_fit_raises(name):
    from forecast_os.core.exceptions import ForecastOSError

    model = get_model(name)
    with pytest.raises(ForecastOSError):
        model.predict(3)
