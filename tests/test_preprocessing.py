"""Tests for preprocessing transforms, calendar features, and Pipeline."""

import numpy as np
import pandas as pd
import pytest

from forecast_os.core.exceptions import ForecastOSError, NotFittedError
from forecast_os.preprocessing.calendar import calendar_features, fourier_features
from forecast_os.preprocessing.pipeline import Pipeline
from forecast_os.preprocessing.transforms import (
    Differencer,
    Imputer,
    LogTransform,
    StandardScaler,
)


def make_panel():
    """Two small numeric-ds series with known values."""
    return pd.DataFrame(
        {
            "unique_id": ["a"] * 5 + ["b"] * 5,
            "ds": list(range(5)) * 2,
            "y": [1.0, 2.0, 4.0, 7.0, 11.0, 100.0, 90.0, 80.0, 70.0, 60.0],
        }
    )


def make_nan_panel():
    return pd.DataFrame(
        {
            "unique_id": ["a"] * 5 + ["b"] * 5,
            "ds": list(range(5)) * 2,
            "y": [np.nan, 2.0, np.nan, 4.0, 5.0, 10.0, np.nan, np.nan, 40.0, np.nan],
        }
    )


# -- Imputer -----------------------------------------------------------------


@pytest.mark.parametrize("method", ["interpolate", "ffill", "mean"])
def test_imputer_fills_all_nan(method):
    out = Imputer(method=method).fit_transform(make_nan_panel())
    assert not out["y"].isna().any()
    assert len(out) == 10


def test_imputer_interpolate_values():
    out = Imputer(method="interpolate").fit_transform(make_nan_panel())
    a = out[out["unique_id"] == "a"]["y"].to_numpy()
    # interior NaN between 2 and 4 -> 3; leading NaN backfilled from 2
    assert a[2] == pytest.approx(3.0)
    assert a[0] == pytest.approx(2.0)


def test_imputer_mean_values():
    out = Imputer(method="mean").fit_transform(make_nan_panel())
    b = out[out["unique_id"] == "b"]["y"].to_numpy()
    assert b[1] == pytest.approx(25.0)  # mean of [10, 40]


def test_imputer_unknown_method_raises():
    with pytest.raises(ValueError):
        Imputer(method="magic")


def test_imputer_inverse_is_identity():
    df = make_panel()
    imp = Imputer().fit(df)
    pd.testing.assert_frame_equal(imp.inverse_transform(df), df)


# -- StandardScaler ----------------------------------------------------------


def test_scaler_round_trip():
    df = make_panel()
    sc = StandardScaler()
    z = sc.fit_transform(df)
    back = sc.inverse_transform(z)
    np.testing.assert_allclose(back["y"].to_numpy(), df["y"].to_numpy(), rtol=1e-10)


def test_scaler_standardizes_per_series():
    z = StandardScaler().fit_transform(make_panel())
    for uid in ("a", "b"):
        g = z[z["unique_id"] == uid]["y"].to_numpy()
        assert abs(g.mean()) < 1e-10
        assert g.std() == pytest.approx(1.0, rel=1e-8)


def test_scaler_forecast_frame_inverse():
    df = make_panel()
    sc = StandardScaler().fit(df)
    ya = df[df["unique_id"] == "a"]["y"].to_numpy()
    mu, sd = ya.mean(), ya.std()
    fc = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": [5, 6],
            "yhat": [0.0, 1.0],
            "lo-80": [-1.0, 0.0],
            "hi-80": [1.0, 2.0],
            "SES-lo-80": [-2.0, -2.0],
            "cutoff": [4, 4],
        }
    )
    inv = sc.inverse_transform(fc)
    np.testing.assert_allclose(inv["yhat"].to_numpy(), [mu, mu + sd])
    np.testing.assert_allclose(inv["lo-80"].to_numpy(), [mu - sd, mu])
    np.testing.assert_allclose(inv["hi-80"].to_numpy(), [mu + sd, mu + 2 * sd])
    np.testing.assert_allclose(inv["SES-lo-80"].to_numpy(), [mu - 2 * sd] * 2)
    # non-value columns untouched
    assert (inv["cutoff"] == 4).all()


def test_scaler_unseen_uid_inverse_raises():
    sc = StandardScaler().fit(make_panel())
    fc = pd.DataFrame({"unique_id": ["zzz"], "ds": [5], "yhat": [0.0]})
    with pytest.raises(ForecastOSError):
        sc.inverse_transform(fc)


def test_scaler_transform_before_fit_raises():
    with pytest.raises(NotFittedError):
        StandardScaler().transform(make_panel())


# -- LogTransform ------------------------------------------------------------


def test_log_round_trip_positive():
    df = make_panel()
    lt = LogTransform()
    back = lt.inverse_transform(lt.fit_transform(df))
    np.testing.assert_allclose(back["y"].to_numpy(), df["y"].to_numpy(), rtol=1e-10)


def test_log_auto_offset_handles_nonpositive():
    df = make_panel()
    df.loc[df["unique_id"] == "a", "y"] = [-3.0, -1.0, 0.0, 2.0, 5.0]
    lt = LogTransform(offset="auto")
    z = lt.fit_transform(df)
    assert np.isfinite(z["y"]).all()
    back = lt.inverse_transform(z)
    np.testing.assert_allclose(back["y"].to_numpy(), df["y"].to_numpy(), atol=1e-10)


def test_log_numeric_offset_and_forecast_frame():
    df = make_panel()
    lt = LogTransform(offset=1.0).fit(df)
    fc = pd.DataFrame({"unique_id": ["a"], "ds": [5], "yhat": [np.log(12.0)]})
    inv = lt.inverse_transform(fc)
    assert inv["yhat"].iloc[0] == pytest.approx(11.0)


def test_log_nonpositive_with_zero_offset_raises():
    df = make_panel()
    df.loc[0, "y"] = -5.0
    with pytest.raises(ForecastOSError):
        LogTransform(offset=0.0).fit_transform(df)


# -- Differencer -------------------------------------------------------------


def test_differencer_transform_values_and_shape():
    df = make_panel()
    z = Differencer(d=1).fit_transform(df)
    assert len(z) == 8  # dropped first row per series
    a = z[z["unique_id"] == "a"]["y"].to_numpy()
    np.testing.assert_allclose(a, [1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize("d", [1, 2])
def test_differencer_in_sample_round_trip(d):
    df = make_panel()
    diff = Differencer(d=d)
    z = diff.fit_transform(df)
    back = diff.inverse_transform(z)
    expected = np.concatenate(
        [df[df["unique_id"] == uid]["y"].to_numpy()[d:] for uid in ("a", "b")]
    )
    np.testing.assert_allclose(back["y"].to_numpy(), expected, rtol=1e-10)


def test_differencer_continuation_inverse():
    df = make_panel()
    diff = Differencer(d=1).fit(df)
    fc = pd.DataFrame({"unique_id": ["a", "a"], "ds": [5, 6], "yhat": [5.0, 6.0]})
    inv = diff.inverse_transform(fc)
    # last y of series a is 11 -> 11+5=16 -> 16+6=22
    np.testing.assert_allclose(inv["yhat"].to_numpy(), [16.0, 22.0])


def test_differencer_continuation_inverse_datetime_ds():
    df = dt_panel().iloc[:5].copy()
    df["y"] = [1.0, 2.0, 4.0, 7.0, 11.0]
    diff = Differencer(d=1).fit(df)
    fc = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": pd.date_range("2024-01-06", periods=2, freq="D"),
            "yhat": [5.0, 6.0],
        }
    )
    inv = diff.inverse_transform(fc)
    np.testing.assert_allclose(inv["yhat"].to_numpy(), [16.0, 22.0])


def test_differencer_too_short_series_raises():
    df = pd.DataFrame({"unique_id": ["a"], "ds": [0], "y": [1.0]})
    with pytest.raises(ForecastOSError):
        Differencer(d=1).fit_transform(df)


# -- calendar features -------------------------------------------------------


def dt_panel():
    return pd.DataFrame(
        {
            "unique_id": "a",
            "ds": pd.date_range("2024-01-01", periods=21, freq="D"),  # a Monday
            "y": np.arange(21, dtype=float),
        }
    )


def test_calendar_raw_columns():
    out = calendar_features(dt_panel(), features=("dayofweek", "month"), cyclical=False)
    assert out["dayofweek"].iloc[0] == 0  # 2024-01-01 is a Monday
    assert (out["month"] == 1).all()


def test_calendar_cyclical_columns():
    out = calendar_features(dt_panel(), features=("dayofweek",), cyclical=True)
    assert "dayofweek_sin" in out.columns and "dayofweek_cos" in out.columns
    assert "dayofweek" not in out.columns
    assert out["dayofweek_sin"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert out["dayofweek_cos"].iloc[0] == pytest.approx(1.0)


def test_calendar_requires_datetime_ds():
    with pytest.raises(ForecastOSError):
        calendar_features(make_panel())


def test_calendar_unknown_feature_raises():
    with pytest.raises(ValueError):
        calendar_features(dt_panel(), features=("lunar_phase",))


def test_fourier_columns_and_periodicity():
    out = fourier_features(dt_panel(), season_length=7, k=2)
    for i in (1, 2):
        assert f"fourier_s7_sin{i}" in out.columns
        assert f"fourier_s7_cos{i}" in out.columns
    s = out["fourier_s7_sin1"].to_numpy()
    np.testing.assert_allclose(s[:14], s[7:21], atol=1e-12)


def test_fourier_per_series_position():
    df = pd.concat([dt_panel(), dt_panel().assign(unique_id="b")], ignore_index=True)
    out = fourier_features(df, season_length=7, k=1)
    a = out[out["unique_id"] == "a"]["fourier_s7_sin1"].to_numpy()
    b = out[out["unique_id"] == "b"]["fourier_s7_sin1"].to_numpy()
    np.testing.assert_allclose(a, b)


# -- Pipeline ----------------------------------------------------------------


def test_pipeline_round_trip_and_order():
    df = make_panel()
    pipe = Pipeline([("log", LogTransform()), ("scale", StandardScaler())])
    z = pipe.fit_transform(df)
    # order: log applied first, then scaling -> per-series mean ~ 0
    for uid in ("a", "b"):
        assert abs(z[z["unique_id"] == uid]["y"].mean()) < 1e-10
    back = pipe.inverse_transform(z)
    np.testing.assert_allclose(back["y"].to_numpy(), df["y"].to_numpy(), rtol=1e-8)


def test_pipeline_named_steps():
    log, sc = LogTransform(), StandardScaler()
    pipe = Pipeline([("log", log), ("scale", sc)])
    assert pipe.named_steps["log"] is log
    assert pipe.named_steps["scale"] is sc


def test_pipeline_duplicate_names_raise():
    with pytest.raises(ValueError):
        Pipeline([("t", LogTransform()), ("t", StandardScaler())])


def test_pipeline_impute_then_scale():
    pipe = Pipeline([("impute", Imputer()), ("scale", StandardScaler())])
    z = pipe.fit_transform(make_nan_panel())
    assert not z["y"].isna().any()


def test_pipeline_transform_matches_fit_transform():
    df = make_panel()
    pipe = Pipeline([("log", LogTransform()), ("scale", StandardScaler())])
    z1 = pipe.fit_transform(df)
    z2 = pipe.transform(df)
    np.testing.assert_allclose(z1["y"].to_numpy(), z2["y"].to_numpy(), rtol=1e-10)
