"""Tests for model persistence: BaseForecaster.save and forecast_os.core.base.load.

Models are saved as a pickle envelope dict carrying the package version, the
class path, and the registry name so files stay self-describing.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

import forecast_os
from forecast_os.core.base import load
from forecast_os.core.exceptions import ForecastOSError
from forecast_os.core.registry import get_model
from forecast_os.core.types import ID_COL, TIME_COL
from forecast_os.datasets.synthetic import generate_series
from forecast_os.models.ml import RidgeLag

H = 6

# One model per major family: baseline, statistical, ml.
REPRESENTATIVE = ["naive", "holt_winters", "ridge_lag"]


def _panel():
    return generate_series(
        n_series=2, length=90, freq="D", trend=0.4, seasonality=7, season_amp=6.0,
        noise=0.8, seed=17,
    )


@pytest.mark.parametrize("name", REPRESENTATIVE)
def test_save_load_roundtrip_identical_predictions(name, tmp_path):
    model = get_model(name).fit(_panel())
    expected = model.predict(H, level=[80])

    path = tmp_path / f"{name}.pkl"
    model.save(path)
    loaded = load(path)

    assert type(loaded) is type(model)
    pd.testing.assert_frame_equal(loaded.predict(H, level=[80]), expected)


def test_envelope_fields(tmp_path):
    model = get_model("naive").fit(_panel())
    path = tmp_path / "naive.pkl"
    model.save(path)

    with open(path, "rb") as f:
        payload = pickle.load(f)
    assert set(payload) == {"forecast_os_version", "class", "registry_name", "model"}
    assert payload["forecast_os_version"] == forecast_os.__version__
    assert payload["class"].endswith("Naive")
    assert payload["registry_name"] == "naive"


def test_load_warns_on_version_mismatch(tmp_path):
    model = get_model("naive").fit(_panel())
    path = tmp_path / "naive.pkl"
    model.save(path)

    with open(path, "rb") as f:
        payload = pickle.load(f)
    payload["forecast_os_version"] = "0.0.0-other"
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    with pytest.warns(UserWarning, match="0.0.0-other"):
        loaded = load(path)
    assert len(loaded.predict(3)) == 2 * 3


def test_load_rejects_non_forecast_os_pickle(tmp_path):
    path = tmp_path / "junk.pkl"
    with open(path, "wb") as f:
        pickle.dump({"hello": "world"}, f)
    with pytest.raises(ForecastOSError, match="not a forecast-os model"):
        load(path)


def test_load_garbage_bytes_raises_forecast_os_error(tmp_path):
    path = tmp_path / "garbage.pkl"
    path.write_bytes(b"this is not a pickle at all \x00\x01\x02")
    with pytest.raises(ForecastOSError, match="not a forecast-os model file"):
        load(path)


def test_load_truncated_envelope_raises_forecast_os_error(tmp_path):
    model = get_model("naive").fit(_panel())
    path = tmp_path / "naive.pkl"
    model.save(path)
    data = path.read_bytes()
    truncated = tmp_path / "truncated.pkl"
    truncated.write_bytes(data[: len(data) // 2])
    with pytest.raises(ForecastOSError, match="not a forecast-os model file"):
        load(truncated)


def test_save_accepts_str_path(tmp_path):
    model = get_model("naive").fit(_panel())
    path = str(tmp_path / "naive-str.pkl")
    model.save(path)
    assert type(load(path)) is type(model)


def test_unfitted_model_roundtrips_and_fits_after_load(tmp_path):
    model = RidgeLag(lags=7, alpha=0.5)
    path = tmp_path / "unfitted.pkl"
    model.save(path)
    loaded = load(path)
    assert loaded.get_params() == model.get_params()
    pred = loaded.fit(_panel()).predict(H)
    assert np.isfinite(pred["yhat"]).all()


def test_exog_fitted_model_roundtrips_with_x_df(tmp_path):
    rng = np.random.default_rng(5)
    n, h = 80, 5
    spend = rng.uniform(0, 10, n)
    df = pd.DataFrame(
        {
            ID_COL: "s0",
            TIME_COL: pd.date_range("2024-01-01", periods=n, freq="D"),
            "y": 10.0 + 2.0 * spend + 0.3 * rng.standard_normal(n),
            "spend": spend,
        }
    )
    train, test = df.iloc[:-h], df.iloc[-h:]
    model = RidgeLag(lags=5).fit(train)
    x_df = test[[ID_COL, TIME_COL, "spend"]]
    expected = model.predict(h, X_df=x_df)

    path = tmp_path / "ridge-exog.pkl"
    model.save(path)
    loaded = load(path)
    pd.testing.assert_frame_equal(loaded.predict(h, X_df=x_df), expected)
