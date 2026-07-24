"""Tests for preprocessing transforms, calendar features, and Pipeline."""

import numpy as np
import pandas as pd
import pytest

from forecast_os.core.exceptions import ForecastOSError, NotFittedError
from forecast_os.core.types import validate_panel
from forecast_os.models.baselines import SeasonalNaive
from forecast_os.preprocessing.calendar import calendar_features, fourier_features
from forecast_os.preprocessing.pipeline import Pipeline
from forecast_os.preprocessing.transforms import (
    Differencer,
    Imputer,
    LogitTransform,
    LogTransform,
    StandardScaler,
    fill_gaps,
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


def test_scaler_cv_frame_inverse_restores_renamed_point_columns():
    """CV frames rename the point column after the model; the inverse must
    restore ALL numeric value columns, not only y/yhat/interval patterns."""
    df = make_panel()
    sc = StandardScaler().fit(df)
    ya = df[df["unique_id"] == "a"]["y"].to_numpy()
    mu, sd = ya.mean(), ya.std()
    fc = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": [5, 6],
            "cutoff": [4, 4],
            "y": [0.5, -0.5],
            "SES": [0.0, 1.0],
            "SES-lo-80": [-1.0, 0.0],
            "SES-hi-80": [1.0, 2.0],
        }
    )
    inv = sc.inverse_transform(fc)
    np.testing.assert_allclose(inv["y"].to_numpy(), [mu + 0.5 * sd, mu - 0.5 * sd])
    np.testing.assert_allclose(inv["SES"].to_numpy(), [mu, mu + sd])
    np.testing.assert_allclose(inv["SES-lo-80"].to_numpy(), [mu - sd, mu])
    np.testing.assert_allclose(inv["SES-hi-80"].to_numpy(), [mu + sd, mu + 2 * sd])
    # identifier columns untouched even though cutoff/ds are numeric
    assert (inv["cutoff"] == 4).all()
    assert list(inv["ds"]) == [5, 6]


def test_inverse_leaves_non_numeric_extra_columns_untouched():
    sc = StandardScaler().fit(make_panel())
    fc = pd.DataFrame(
        {"unique_id": ["a"], "ds": [5], "yhat": [0.0], "note": ["keep-me"]}
    )
    inv = sc.inverse_transform(fc)
    assert inv["note"].iloc[0] == "keep-me"


def test_forward_transform_ignores_renamed_point_columns():
    # the widened column selection is inverse-only; forward must not touch
    # arbitrary passthrough columns
    df = make_panel().assign(SES=1.0)
    z = StandardScaler().fit(make_panel()).transform(df)
    assert (z["SES"] == 1.0).all()


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


def test_log_inverse_restores_renamed_point_column():
    df = make_panel()
    lt = LogTransform(offset=1.0).fit(df)
    fc = pd.DataFrame({"unique_id": ["a"], "ds": [5], "SES": [np.log(12.0)]})
    inv = lt.inverse_transform(fc)
    assert inv["SES"].iloc[0] == pytest.approx(11.0)


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


def test_differencer_continuation_inverse_renamed_columns():
    df = make_panel()
    diff = Differencer(d=1).fit(df)
    fc = pd.DataFrame(
        {"unique_id": ["a", "a"], "ds": [5, 6], "cutoff": [4, 4], "SES": [5.0, 6.0]}
    )
    inv = diff.inverse_transform(fc)
    np.testing.assert_allclose(inv["SES"].to_numpy(), [16.0, 22.0])
    assert (inv["cutoff"] == 4).all()


def test_differencer_partial_in_range_slice_raises():
    df = make_panel()
    diff = Differencer(d=1)
    z = diff.fit_transform(df)
    partial = z.iloc[1:].copy()  # drop the first transformed row of series a
    with pytest.raises(ForecastOSError, match="not invertible"):
        diff.inverse_transform(partial)


def test_differencer_shifted_in_range_slice_raises():
    df = make_panel()
    diff = Differencer(d=1).fit(df)
    # correct length (n - d = 4) but starting at the training start, not the
    # first transformed ds -> silently-wrong values before, now an error
    fc = pd.DataFrame({"unique_id": ["a"] * 4, "ds": [0, 1, 2, 3], "yhat": [1.0] * 4})
    with pytest.raises(ForecastOSError, match="not invertible"):
        diff.inverse_transform(fc)


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


# -- LogitTransform ----------------------------------------------------------


def rate_panel():
    """Two series of rates strictly inside (0, 1)."""
    return pd.DataFrame(
        {
            "unique_id": ["a"] * 5 + ["b"] * 5,
            "ds": list(range(5)) * 2,
            "y": [0.1, 0.2, 0.5, 0.7, 0.9, 0.02, 0.4, 0.55, 0.6, 0.98],
        }
    )


def test_logit_round_trip_within_bounds():
    df = rate_panel()
    lt = LogitTransform()
    back = lt.inverse_transform(lt.fit_transform(df))
    np.testing.assert_allclose(back["y"].to_numpy(), df["y"].to_numpy(), rtol=1e-9)


def test_logit_transform_values():
    df = rate_panel()
    z = LogitTransform().fit_transform(df)
    y = df["y"].to_numpy()
    np.testing.assert_allclose(z["y"].to_numpy(), np.log(y / (1 - y)), rtol=1e-12)


def test_logit_custom_bounds_round_trip():
    df = rate_panel()
    df["y"] = df["y"] * 100.0  # rates in (0, 100)
    lt = LogitTransform(lower=0.0, upper=100.0)
    back = lt.inverse_transform(lt.fit_transform(df))
    np.testing.assert_allclose(back["y"].to_numpy(), df["y"].to_numpy(), rtol=1e-9)


def test_logit_boundary_values_squeezed_finite():
    df = rate_panel()
    df.loc[0, "y"] = 0.0
    df.loc[4, "y"] = 1.0
    lt = LogitTransform(eps=1e-6)
    z = lt.fit_transform(df)
    assert np.isfinite(z["y"]).all()
    back = lt.inverse_transform(z)
    assert (back["y"] > 0.0).all() and (back["y"] < 1.0).all()


def test_logit_out_of_bounds_raises_at_fit():
    for bad in (1.5, -0.1):
        df = rate_panel()
        df.loc[2, "y"] = bad
        with pytest.raises(ForecastOSError):
            LogitTransform().fit(df)


def test_logit_forecast_frame_intervals_stay_inside_bounds():
    lt = LogitTransform(lower=0.0, upper=1.0).fit(rate_panel())
    fc = pd.DataFrame(
        {
            "unique_id": ["a"] * 3,
            "ds": [5, 6, 7],
            "yhat": [0.0, 5.0, -5.0],
            "lo-80": [-30.0, -1.0, -40.0],
            "hi-80": [30.0, 8.0, 2.0],
        }
    )
    inv = lt.inverse_transform(fc)
    for col in ("yhat", "lo-80", "hi-80"):
        assert (inv[col] > 0.0).all() and (inv[col] < 1.0).all()
    assert (inv["lo-80"] < inv["yhat"]).all() and (inv["yhat"] < inv["hi-80"]).all()
    assert inv["yhat"].iloc[0] == pytest.approx(0.5)


def test_logit_inverse_huge_logit_stays_strictly_inside_bounds():
    """expit saturates to exactly 1.0 for x > ~36.7 (and 0.0 for very negative
    x); the inverse must clamp so results never land ON a bound."""
    lt = LogitTransform(lower=0.0, upper=1.0).fit(rate_panel())
    fc = pd.DataFrame({"unique_id": ["a"] * 2, "ds": [5, 6], "yhat": [1000.0, -1000.0]})
    inv = lt.inverse_transform(fc)
    assert inv["yhat"].iloc[0] < 1.0  # strictly below the upper bound
    assert inv["yhat"].iloc[1] > 0.0  # strictly above the lower bound

    df = rate_panel()
    df["y"] = df["y"] * 100.0
    lt = LogitTransform(lower=0.0, upper=100.0).fit(df)
    inv = lt.inverse_transform(fc)
    assert inv["yhat"].iloc[0] < 100.0
    assert inv["yhat"].iloc[1] > 0.0


def test_logit_per_series_dict_bounds():
    df = rate_panel()
    df.loc[df["unique_id"] == "b", "y"] = [2.0, 40.0, 55.0, 60.0, 98.0]
    lt = LogitTransform(lower=0.0, upper={"a": 1.0, "b": 100.0})
    z = lt.fit_transform(df)
    back = lt.inverse_transform(z)
    np.testing.assert_allclose(back["y"].to_numpy(), df["y"].to_numpy(), rtol=1e-9)
    b = z[z["unique_id"] == "b"]["y"].to_numpy()
    np.testing.assert_allclose(b[1], np.log(40.0 / 60.0), rtol=1e-12)


def test_logit_dict_bounds_unseen_series_raises():
    lt = LogitTransform(upper={"a": 1.0})
    with pytest.raises(ForecastOSError):
        lt.fit(rate_panel())  # series "b" has no bound


def test_logit_invalid_constructor():
    with pytest.raises(ValueError):
        LogitTransform(lower=1.0, upper=1.0)
    with pytest.raises(ValueError):
        LogitTransform(lower=2.0, upper=1.0)
    for bad_eps in (0.0, -1e-6, 0.5):
        with pytest.raises(ValueError):
            LogitTransform(eps=bad_eps)


def test_logit_transform_before_fit_raises():
    with pytest.raises(NotFittedError):
        LogitTransform().transform(rate_panel())


# -- fill_gaps ---------------------------------------------------------------


def gapped_weekly_panel():
    """12 Mondays of period-4 demand with one week (a zero) missing.

    The gap sits inside the final season so positional models misalign.
    """
    ds = pd.date_range("2025-01-06", periods=12, freq="W-MON")
    y = [10.0, 0.0, 0.0, 0.0] * 3
    df = pd.DataFrame({"unique_id": "sku", "ds": ds, "y": y})
    return df.drop(index=9).reset_index(drop=True)


def test_fill_gaps_restores_regular_grid():
    out = fill_gaps(gapped_weekly_panel(), freq="W-MON")
    assert len(out) == 12
    validate_panel(out)  # must be a valid panel
    assert pd.infer_freq(out["ds"]) == "W-MON"
    assert out["y"].iloc[9] == 0.0
    # existing observations untouched
    np.testing.assert_allclose(out["y"].to_numpy(), [10.0, 0.0, 0.0, 0.0] * 3)


def test_fill_gaps_seasonal_alignment_regression():
    """The gap-analysis repro: a missing week shifts every later observation
    one position left, so a seasonal model reads the wrong phase. fill_gaps
    must restore positional ds alignment."""
    df = gapped_weekly_panel()
    misaligned = SeasonalNaive(season_length=4).fit(df).predict(4)["yhat"].to_numpy()
    filled = fill_gaps(df, freq="W-MON")
    aligned = SeasonalNaive(season_length=4).fit(filled).predict(4)["yhat"].to_numpy()
    np.testing.assert_allclose(aligned, [10.0, 0.0, 0.0, 0.0])
    assert not np.allclose(misaligned, aligned)


def test_fill_gaps_extra_columns_nan_on_inserted_rows():
    df = gapped_weekly_panel().assign(promo=1.0)
    out = fill_gaps(df, freq="W-MON")
    assert len(out) == 12
    assert np.isnan(out["promo"].iloc[9])
    assert (out["promo"].drop(index=9) == 1.0).all()


def test_fill_gaps_nan_fill_value():
    out = fill_gaps(gapped_weekly_panel(), freq="W-MON", fill_value=np.nan)
    assert np.isnan(out["y"].iloc[9])
    imputed = Imputer(method="ffill").fit_transform(out)
    assert not imputed["y"].isna().any()


def test_fill_gaps_per_series_ranges_independent():
    a = pd.DataFrame(
        {
            "unique_id": "a",
            "ds": pd.to_datetime(["2025-01-01", "2025-01-03"]),
            "y": [1.0, 2.0],
        }
    )
    b = pd.DataFrame(
        {
            "unique_id": "b",
            "ds": pd.to_datetime(["2025-02-01", "2025-02-02"]),
            "y": [3.0, 4.0],
        }
    )
    out = fill_gaps(pd.concat([a, b], ignore_index=True), freq="D")
    assert len(out[out["unique_id"] == "a"]) == 3  # one gap day inserted
    assert len(out[out["unique_id"] == "b"]) == 2  # b is not padded to a's range
    assert out[out["unique_id"] == "a"]["y"].iloc[1] == 0.0


def test_fill_gaps_numeric_ds_raises():
    with pytest.raises(ForecastOSError):
        fill_gaps(make_panel(), freq="D")


def test_fill_gaps_off_grid_ds_raises():
    df = gapped_weekly_panel()
    df.loc[3, "ds"] = df.loc[3, "ds"] + pd.Timedelta(days=1)  # a Tuesday
    with pytest.raises(ForecastOSError):
        fill_gaps(df, freq="W-MON")
