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
from forecast_os.core.types import ID_COL, TIME_COL
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


# Models whose fit() contract restricts inputs to a domain (e.g. retention
# fractions in [0, 1]) cannot fit the generic panel; they carry their own
# contract-equivalent tests next to their implementation.
_DOMAIN_PANEL_MODELS = {"retention_sbg"}  # see tests/test_gtm_retention.py


def _skip_if_domain_restricted(name):
    if name in _DOMAIN_PANEL_MODELS:
        pytest.skip(f"{name} has a domain-restricted input contract; covered by its own tests")


@pytest.mark.parametrize("name", _all_model_names())
def test_registered_model_contract(name):
    _skip_if_domain_restricted(name)
    model = get_model(name)
    assert isinstance(model, BaseForecaster)

    df = _contract_panel()
    fitted = model.fit(df)
    assert fitted is model, "fit() must return self"

    pred = model.predict(H, level=[80])
    assert list(pred.columns[:3]) == [ID_COL, TIME_COL, "yhat"]
    assert {"lo-80", "hi-80"} <= set(pred.columns)

    # h rows per series, all finite point forecasts. Every training series must
    # be forecast; hierarchical models may add aggregate series (e.g. "total").
    counts = pred.groupby(ID_COL).size()
    assert set(counts.index) >= set(df[ID_COL].unique())
    assert (counts == H).all()
    assert np.isfinite(pred["yhat"]).all(), f"{name} produced non-finite forecasts"
    assert (pred["lo-80"] <= pred["hi-80"] + 1e-9).all()

    # future ds strictly after the training data, strictly increasing.
    # Aggregate series not present in training are checked against the panel max.
    last_train = df.groupby(ID_COL)[TIME_COL].max()
    for uid, g in pred.groupby(ID_COL):
        assert (g[TIME_COL] > last_train.get(uid, last_train.max())).all()
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


# -- negative predict-argument tests -------------------------------------------
# Fitted models are cached per module run so each registered model is fitted
# once and reused across all parametrized violations (keeps runtime low).

_FITTED_CACHE: dict[str, BaseForecaster] = {}


def _fitted_model(name: str) -> BaseForecaster:
    _skip_if_domain_restricted(name)
    if name not in _FITTED_CACHE:
        panel = generate_series(
            n_series=1, length=40, freq="D", trend=0.2, seasonality=7,
            season_amp=3.0, noise=0.5, seed=7,
        )
        _FITTED_CACHE[name] = get_model(name).fit(panel)
    return _FITTED_CACHE[name]


@pytest.mark.parametrize("bad_h", [0, -1])
@pytest.mark.parametrize("name", _all_model_names())
def test_registered_model_predict_rejects_nonpositive_h(name, bad_h):
    model = _fitted_model(name)
    with pytest.raises(ValueError):
        model.predict(bad_h)


@pytest.mark.parametrize("bad_level", [[0], [150]])
@pytest.mark.parametrize("name", _all_model_names())
def test_registered_model_predict_rejects_bad_level(name, bad_level):
    model = _fitted_model(name)
    with pytest.raises(ValueError):
        model.predict(3, level=bad_level)


# -- persistence ----------------------------------------------------------------


@pytest.mark.parametrize("name", _all_model_names())
def test_registered_model_save_load_roundtrip(name, tmp_path):
    from forecast_os.core.base import load

    _skip_if_domain_restricted(name)
    model = get_model(name).fit(_contract_panel())
    expected = model.predict(H, level=[80])

    path = tmp_path / f"{name}.pkl"
    model.save(path)
    loaded = load(path)

    assert type(loaded) is type(model)
    pd.testing.assert_frame_equal(loaded.predict(H, level=[80]), expected)
