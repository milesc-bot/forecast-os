"""Hand-computed tests for evaluation.metrics."""

import warnings

import numpy as np
import pandas as pd
import pytest

from forecast_os.evaluation.metrics import (
    bias,
    coverage,
    evaluate,
    mae,
    mape,
    mase,
    pct_bias,
    pinball_loss,
    rmse,
    rmsse,
    smape,
    tracking_signal,
    winkler_score,
)

# -- point metrics ------------------------------------------------------------


def test_mae_hand_computed():
    assert mae([1, 2], [2, 4]) == pytest.approx(1.5)
    assert mae([1, 2, 3], [1, 2, 3]) == 0.0


def test_rmse_hand_computed():
    assert rmse([1, 2], [2, 4]) == pytest.approx(np.sqrt(2.5))
    assert rmse([5], [5]) == 0.0


def test_mape_excludes_zero_targets():
    # y=0 term dropped; only |(2-3)/2| = 0.5 remains
    assert mape([0, 2], [1, 3]) == pytest.approx(0.5)


def test_mape_all_zero_targets_is_nan():
    assert np.isnan(mape([0, 0], [1, 2]))


def test_mape_is_a_fraction():
    assert mape([100], [110]) == pytest.approx(0.1)


def test_smape_zero_over_zero_term_is_zero():
    assert smape([0, 2], [0, 2]) == 0.0


def test_smape_hand_computed():
    # |1-3| / ((1+3)/2) = 1.0
    assert smape([1], [3]) == pytest.approx(1.0)
    # 0 actual vs positive forecast hits the upper bound of 2
    assert smape([0], [5]) == pytest.approx(2.0)


def test_smape_bounded():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(50)
    yhat = rng.standard_normal(50)
    assert 0.0 <= smape(y, yhat) <= 2.0


# -- governance (bias) metrics ------------------------------------------------


def test_bias_hand_computed():
    assert bias([1, 2], [2, 4]) == pytest.approx(1.5)
    assert bias([2, 4], [1, 2]) == pytest.approx(-1.5)
    assert bias([1, 2], [1, 2]) == 0.0


def test_bias_cancels_symmetric_errors():
    # signed: offsetting misses net to zero while mae would report 2
    assert bias([10, 10], [8, 12]) == 0.0


def test_pct_bias_hand_computed():
    assert pct_bias([100, 100], [95, 89]) == pytest.approx(-0.08)
    # negatives count through |y| in the denominator: 10 / (100 + 100)
    assert pct_bias([-100, 100], [-100, 110]) == pytest.approx(0.05)


def test_pct_bias_zero_denominator_is_nan():
    assert np.isnan(pct_bias([0, 0], [1, -1]))


def test_tracking_signal_hand_computed():
    assert tracking_signal([1, 2, 3], [2, 3, 4]) == pytest.approx(3.0)
    assert tracking_signal([2, 3, 4], [1, 2, 3]) == pytest.approx(-3.0)
    # offsetting errors: signed sum 0 over MAD 1
    assert tracking_signal([0, 0], [1, -1]) == pytest.approx(0.0)


def test_tracking_signal_perfect_forecast_is_nan():
    assert np.isnan(tracking_signal([1, 2], [1, 2]))


def test_bias_metrics_expose_sandbagged_forecast():
    """A forecast running 8% low is clearly negative on the bias metrics
    while mape reports the same 0.08 it would for an 8%-high forecast."""
    y = np.array([100.0, 200.0, 300.0, 400.0])
    yhat = 0.92 * y
    assert bias(y, yhat) == pytest.approx(-20.0)
    assert pct_bias(y, yhat) == pytest.approx(-0.08)
    assert tracking_signal(y, yhat) == pytest.approx(-4.0)  # every error negative
    assert mape(y, yhat) == pytest.approx(0.08)  # direction hidden


@pytest.mark.parametrize("fn", [bias, pct_bias, tracking_signal])
def test_bias_metric_shape_mismatch_raises(fn):
    with pytest.raises(ValueError, match="shape mismatch"):
        fn([1, 2, 3], [1, 2])


@pytest.mark.parametrize("fn", [bias, pct_bias, tracking_signal])
def test_bias_metric_empty_input_raises(fn):
    with pytest.raises(ValueError, match="empty"):
        fn([], [])


# -- scaled metrics -----------------------------------------------------------


def test_mase_scale_m1_equals_mae():
    y_train = [1, 2, 3, 4, 5]  # naive m=1 in-sample MAE = 1
    y, yhat = [6, 7], [6.5, 8]
    assert mase(y, yhat, y_train, m=1) == pytest.approx(mae(y, yhat))


def test_mase_seasonal_scale():
    y_train = [1, 2, 3, 4, 5]  # |y[t] - y[t-2]| = 2 everywhere -> scale 2
    y, yhat = [6, 7], [7, 9]
    assert mase(y, yhat, y_train, m=2) == pytest.approx(mae(y, yhat) / 2)


def test_mase_constant_train_is_nan():
    assert np.isnan(mase([1, 2], [1, 2], [3, 3, 3, 3], m=1))


def test_mase_train_too_short_raises():
    with pytest.raises(ValueError, match="longer than m"):
        mase([1], [1], [1, 2], m=2)


def test_rmsse_scale_m1_equals_rmse():
    y_train = [1, 2, 3, 4, 5]  # squared naive scale = 1
    y, yhat = [6, 8], [7, 7]
    assert rmsse(y, yhat, y_train, m=1) == pytest.approx(rmse(y, yhat))


# -- probabilistic metrics ----------------------------------------------------


def test_pinball_at_median_is_half_mae():
    y, q_pred = [1.0, 4.0, 2.0], [2.0, 2.0, 2.0]
    assert pinball_loss(y, q_pred, q=0.5) == pytest.approx(0.5 * mae(y, q_pred))


def test_pinball_asymmetry():
    # under-forecast at q=0.9 costs 0.9 per unit, over-forecast costs 0.1
    assert pinball_loss([2], [1], q=0.9) == pytest.approx(0.9)
    assert pinball_loss([1], [2], q=0.9) == pytest.approx(0.1)


def test_pinball_invalid_q_raises():
    with pytest.raises(ValueError, match="q must be in"):
        pinball_loss([1], [1], q=1.5)


def test_coverage_hand_computed():
    # 1 in [0,2] yes; 2 in [3,4] no; 3 in [2,4] yes
    assert coverage([1, 2, 3], [0, 3, 2], [2, 4, 4]) == pytest.approx(2 / 3)
    assert coverage([1, 2], [0, 0], [9, 9]) == 1.0


def test_winkler_inside_interval_equals_width():
    # both actuals inside their intervals -> score is just the mean width
    assert winkler_score([1, 2], [0, 0], [4, 4], level=80) == pytest.approx(4.0)
    assert winkler_score([5], [4], [7], level=95) == pytest.approx(3.0)


def test_winkler_below_lo_adds_penalty():
    # level=80 -> a=0.2 -> penalty factor 2/a = 10
    # width 3, y=1 undershoots lo=2 by 1: W = 3 + 10*1 = 13
    assert winkler_score([1], [2], [5], level=80) == pytest.approx(13.0)


def test_winkler_above_hi_adds_penalty():
    # width 3, y=7 overshoots hi=5 by 2: W = 3 + 10*2 = 23
    assert winkler_score([7], [2], [5], level=80) == pytest.approx(23.0)


def test_winkler_mixed_mean():
    # element 1 inside (W=4), element 2 below lo by 1 (W = 3 + 10 = 13)
    assert winkler_score([1, 1], [0, 2], [4, 5], level=80) == pytest.approx(8.5)


@pytest.mark.parametrize("level", [0, 100, -5, 150])
def test_winkler_invalid_level_raises(level):
    with pytest.raises(ValueError, match="level must be in"):
        winkler_score([1], [0], [2], level=level)


def test_winkler_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        winkler_score([1, 2], [0], [3, 3], level=80)


# -- input validation ---------------------------------------------------------


@pytest.mark.parametrize("fn", [mae, rmse, mape, smape])
def test_shape_mismatch_raises(fn):
    with pytest.raises(ValueError, match="shape mismatch"):
        fn([1, 2, 3], [1, 2])


@pytest.mark.parametrize("fn", [mae, rmse, mape, smape])
def test_empty_input_raises(fn):
    with pytest.raises(ValueError, match="empty"):
        fn([], [])


def test_coverage_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        coverage([1, 2], [0], [3, 3])


# -- evaluate() ---------------------------------------------------------------


def _cv_frame():
    ds = list(pd.date_range("2024-01-05", periods=2, freq="D"))
    return pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": ds * 2,
            "cutoff": [pd.Timestamp("2024-01-04")] * 4,
            "y": [1.0, 2.0, 3.0, 4.0],
            "m1": [1.5, 2.5, 3.5, 4.5],
            "m1-lo-80": [0.0, 0.0, 0.0, 0.0],
            "m1-hi-80": [9.0, 9.0, 9.0, 9.0],
            "m2": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_evaluate_rows_and_values():
    res = evaluate(_cv_frame(), metrics=("mae", "rmse"))
    # n_series x n_metrics rows
    assert len(res) == 2 * 2
    assert set(res["unique_id"]) == {"a", "b"}
    assert set(res["metric"]) == {"mae", "rmse"}
    row = res[(res["unique_id"] == "a") & (res["metric"] == "mae")]
    assert row["m1"].iloc[0] == pytest.approx(0.5)
    assert row["m2"].iloc[0] == pytest.approx(0.0)


def test_evaluate_excludes_interval_columns_from_models():
    res = evaluate(_cv_frame(), metrics=("mae",))
    assert "m1-lo-80" not in res.columns
    assert "m1-hi-80" not in res.columns
    assert {"m1", "m2"} <= set(res.columns)


def test_evaluate_mase_requires_train_df():
    with pytest.raises(ValueError, match="requires train_df"):
        evaluate(_cv_frame(), metrics=("mase",))


def test_evaluate_mase_with_train_df():
    train = pd.DataFrame(
        {
            "unique_id": ["a"] * 5 + ["b"] * 5,
            "ds": list(pd.date_range("2024-01-01", periods=5)) * 2,
            "y": [1.0, 2.0, 3.0, 4.0, 5.0] * 2,  # naive scale 1 -> mase == mae
        }
    )
    res = evaluate(_cv_frame(), metrics=("mase",), train_df=train, seasonality=1)
    row = res[(res["unique_id"] == "a") & (res["metric"] == "mase")]
    assert row["m1"].iloc[0] == pytest.approx(0.5)


def _train_frame():
    """Two interleaved series whose naive scale is 1.0 when read in ds order."""
    return pd.DataFrame(
        {
            "unique_id": ["a", "b"] * 5,
            "ds": np.repeat(pd.date_range("2024-01-01", periods=5), 2),
            "y": [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0],
        }
    )


@pytest.mark.parametrize("metric, expected", [("mase", 0.5), ("rmsse", 0.5)])
def test_evaluate_scaled_metrics_are_order_invariant(metric, expected):
    """evaluate() read train_df in whatever order the caller passed it, so the
    same panel shuffled produced a silently different MASE/RMSSE (the audit
    measured mase 1.038 sorted vs 0.165 for the identical rows shuffled, with
    no warning). MASE/RMSSE scale on ``mean(|y_t - y_{t-m}|)``, which is only
    defined on the chronologically ordered series, so the sorted value is the
    only correct one. An unsorted panel is legitimate under the library's own
    contract — validate_panel sorts it, and cross_validation is already
    order-invariant — so the same frame must not yield a correct cv_df and a
    wrong metric.
    """
    train = _train_frame()
    shuffled = train.sample(frac=1.0, random_state=7)
    assert not shuffled.index.equals(train.index), "fixture must actually be reordered"

    sorted_res = evaluate(_cv_frame(), metrics=(metric,), train_df=train, seasonality=1)
    shuffled_res = evaluate(
        _cv_frame(), metrics=(metric,), train_df=shuffled, seasonality=1
    )

    # series 'a' is 1..5 in ds order -> naive scale 1.0 -> the metric is the
    # plain mae/rmse of the forecast, 0.5 for model m1.
    row = sorted_res[(sorted_res["unique_id"] == "a") & (sorted_res["metric"] == metric)]
    assert row["m1"].iloc[0] == pytest.approx(expected)
    pd.testing.assert_frame_equal(shuffled_res, sorted_res)


def test_evaluate_scaled_metrics_reject_train_df_without_ds():
    """Sorting train_df needs a 'ds' column. Silently falling back to caller
    order is what produced the wrong scale in the first place, so a train_df
    with no time column must be rejected by name rather than tolerated.
    """
    train = _train_frame().drop(columns=["ds"])
    with pytest.raises(ValueError, match="missing required column 'ds'"):
        evaluate(_cv_frame(), metrics=("mase",), train_df=train, seasonality=1)


def test_evaluate_ignores_train_df_shape_when_no_scaled_metric_requested():
    """train_df is only consumed by mase/rmsse, so passing a ds-less frame
    alongside point metrics stays accepted — the new ordering contract must not
    break callers that hand evaluate() a train_df it never reads.
    """
    train = _train_frame().drop(columns=["ds"])
    res = evaluate(_cv_frame(), metrics=("mae",), train_df=train)
    row = res[(res["unique_id"] == "a") & (res["metric"] == "mae")]
    assert row["m1"].iloc[0] == pytest.approx(0.5)


def test_evaluate_unknown_metric_raises():
    with pytest.raises(ValueError, match="unknown metric"):
        evaluate(_cv_frame(), metrics=("banana",))


def test_evaluate_missing_required_column_raises():
    bad = _cv_frame().drop(columns=["y"])
    with pytest.raises(ValueError, match="missing required column"):
        evaluate(bad)


def test_evaluate_no_model_columns_raises():
    meta_only = _cv_frame()[["unique_id", "ds", "cutoff", "y"]]
    with pytest.raises(ValueError, match="no model forecast columns"):
        evaluate(meta_only)


def test_evaluate_supports_bias_metrics():
    res = evaluate(_cv_frame(), metrics=("bias", "pct_bias", "tracking_signal"))
    row = res[(res["unique_id"] == "a") & (res["metric"] == "bias")]
    assert row["m1"].iloc[0] == pytest.approx(0.5)  # m1 = y + 0.5 everywhere
    row = res[(res["unique_id"] == "a") & (res["metric"] == "pct_bias")]
    assert row["m1"].iloc[0] == pytest.approx(1.0 / 3.0)  # (0.5 + 0.5) / (1 + 2)
    ts = res[(res["unique_id"] == "b") & (res["metric"] == "tracking_signal")]
    assert ts["m1"].iloc[0] == pytest.approx(2.0)  # both errors +0.5
    assert np.isnan(ts["m2"].iloc[0])  # m2 is perfect: MAD ~ 0 -> nan


# -- evaluate(by="cutoff") ------------------------------------------------------


def _two_cutoff_frame():
    """Two series x two cutoffs with hand-computable per-cutoff errors."""
    c1, c2 = pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04")
    ds = list(pd.date_range("2024-01-03", periods=4, freq="D"))
    return pd.DataFrame(
        {
            "unique_id": ["a"] * 4 + ["b"] * 4,
            "ds": ds * 2,
            "cutoff": [c1, c1, c2, c2] * 2,
            "y": [1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0],
            "m1": [2.0, 3.0, 3.0, 7.0, 10.0, 12.0, 10.0, 10.0],
        }
    )


def test_evaluate_by_cutoff_row_counts_and_values():
    c1, c2 = pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04")
    res = evaluate(_two_cutoff_frame(), metrics=("mae", "bias"), by="cutoff")
    # n_series x n_cutoffs x n_metrics rows, with a cutoff column
    assert len(res) == 2 * 2 * 2
    assert "cutoff" in res.columns
    assert set(res["cutoff"]) == {c1, c2}
    mae_rows = res[res["metric"] == "mae"].set_index(["unique_id", "cutoff"])["m1"]
    assert mae_rows[("a", c1)] == pytest.approx(1.0)  # errors +1, +1
    assert mae_rows[("a", c2)] == pytest.approx(1.5)  # errors 0, +3
    assert mae_rows[("b", c1)] == pytest.approx(1.0)  # errors 0, +2
    assert mae_rows[("b", c2)] == pytest.approx(0.0)
    bias_rows = res[res["metric"] == "bias"].set_index(["unique_id", "cutoff"])["m1"]
    assert bias_rows[("a", c2)] == pytest.approx(1.5)


def test_evaluate_default_pools_cutoffs():
    res = evaluate(_two_cutoff_frame(), metrics=("mae",))
    assert len(res) == 2  # one row per series, no cutoff column
    assert "cutoff" not in res.columns
    row = res[res["unique_id"] == "a"]
    assert row["m1"].iloc[0] == pytest.approx(1.25)  # (1 + 1 + 0 + 3) / 4


def test_evaluate_interval_metrics_by_cutoff():
    c1, c2 = pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04")
    df = pd.DataFrame(
        {
            "unique_id": "a",
            "ds": pd.date_range("2024-01-03", periods=4, freq="D"),
            "cutoff": [c1, c1, c2, c2],
            "y": [1.0, 2.0, 3.0, 4.0],
            "m1": [1.0, 2.0, 3.0, 4.0],
            "m1-lo-80": [0.0, 0.0, 4.0, 5.0],
            "m1-hi-80": [2.0, 3.0, 5.0, 6.0],
        }
    )
    res = evaluate(df, metrics=("coverage",), by="cutoff").set_index("cutoff")
    assert list(res["metric"]) == ["coverage-80"] * 2
    assert res.loc[c1, "m1"] == pytest.approx(1.0)  # both actuals inside
    assert res.loc[c2, "m1"] == pytest.approx(0.0)  # both actuals outside


def test_evaluate_unknown_by_raises():
    with pytest.raises(ValueError, match="unknown by"):
        evaluate(_cv_frame(), metrics=("mae",), by="model")


def test_evaluate_by_cutoff_requires_cutoff_column():
    no_cutoff = _cv_frame().drop(columns=["cutoff"])
    with pytest.raises(ValueError, match="cutoff"):
        evaluate(no_cutoff, metrics=("mae",), by="cutoff")


# -- evaluate() interval metrics ------------------------------------------------


def _interval_cv_frame():
    """One series, one model, known 80% intervals for hand-computed scores."""
    return pd.DataFrame(
        {
            "unique_id": ["a"] * 4,
            "ds": list(pd.date_range("2024-01-05", periods=4, freq="D")),
            "cutoff": [pd.Timestamp("2024-01-04")] * 4,
            "y": [1.0, 2.0, 3.0, 4.0],
            "m1": [1.5, 2.5, 3.5, 4.5],
            "m1-lo-80": [0.0, 3.0, 2.0, 5.0],
            "m1-hi-80": [2.0, 4.0, 4.0, 6.0],
        }
    )


def test_evaluate_coverage_and_winkler_rows():
    res = evaluate(_interval_cv_frame(), metrics=("coverage", "winkler"))
    assert set(res["metric"]) == {"coverage-80", "winkler-80"}
    # y in [lo, hi]: yes, no, yes, no -> 0.5
    cov = res[res["metric"] == "coverage-80"]
    assert cov["m1"].iloc[0] == pytest.approx(0.5)
    # widths [2,1,2,1]; misses undershoot lo by 1 -> +10 each (a=0.2)
    # W = [2, 11, 2, 11] -> mean 6.5
    wink = res[res["metric"] == "winkler-80"]
    assert wink["m1"].iloc[0] == pytest.approx(6.5)


def test_evaluate_pinball_rows():
    res = evaluate(_interval_cv_frame(), metrics=("pinball",))
    assert list(res["metric"]) == ["pinball-80"]
    # q=0.1 on lo: per-element [0.1, 0.9, 0.1, 0.9] -> 0.5
    # q=0.9 on hi: per-element [0.1, 0.2, 0.1, 0.2] -> 0.15
    assert res["m1"].iloc[0] == pytest.approx(0.5 * (0.5 + 0.15))


def _degenerate_interval_frame():
    """lo == yhat == hi at every level: wis collapses to point-forecast MAE."""
    df = pd.DataFrame(
        {
            "unique_id": ["a"] * 4,
            "ds": list(pd.date_range("2024-01-05", periods=4, freq="D")),
            "cutoff": [pd.Timestamp("2024-01-04")] * 4,
            "y": [1.0, 2.0, 3.0, 4.0],
            "m1": [2.0, 2.0, 2.0, 2.0],
            "m2": [1.0, 2.0, 3.0, 4.0],  # perfect forecast
        }
    )
    for model in ("m1", "m2"):
        for lvl in (80, 95):
            df[f"{model}-lo-{lvl}"] = df[model]
            df[f"{model}-hi-{lvl}"] = df[model]
    return df


def test_evaluate_wis_degenerate_equals_point_pinball_mean():
    df = _degenerate_interval_frame()
    res = evaluate(df, metrics=("wis",))
    assert list(res["metric"]) == ["wis"]
    # implied quantiles of levels {80, 95}: 0.1/0.9 and 0.025/0.975
    y, yhat = df["y"], df["m1"]
    expected = np.mean(
        [2 * pinball_loss(y, yhat, q) for q in (0.1, 0.9, 0.025, 0.975)]
    )
    assert res["m1"].iloc[0] == pytest.approx(expected)
    # degenerate intervals make wis the point MAE (here 1.0)
    assert res["m1"].iloc[0] == pytest.approx(mae(y, yhat))
    # a forecast equal to y scores exactly 0
    assert res["m2"].iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_evaluate_interval_metric_without_columns_raises():
    # _cv_frame has intervals for m1 but none for m2 -> helpful error
    with pytest.raises(ValueError, match=r"level=\["):
        evaluate(_cv_frame(), metrics=("coverage",))
    no_intervals = _interval_cv_frame()[["unique_id", "ds", "cutoff", "y", "m1"]]
    with pytest.raises(ValueError, match="cross_validation"):
        evaluate(no_intervals, metrics=("winkler",))


def test_evaluate_mixes_point_and_interval_metrics():
    res = evaluate(_interval_cv_frame(), metrics=("mae", "coverage"))
    assert set(res["metric"]) == {"mae", "coverage-80"}
    assert res[res["metric"] == "mae"]["m1"].iloc[0] == pytest.approx(0.5)


def test_evaluate_interval_level_union_nan_for_missing_level():
    df = _interval_cv_frame()
    df["m2"] = df["y"]
    df["m2-lo-80"] = df["y"] - 1.0
    df["m2-hi-80"] = df["y"] + 1.0
    df["m2-lo-95"] = df["y"] - 2.0
    df["m2-hi-95"] = df["y"] + 2.0
    res = evaluate(df, metrics=("coverage",))
    assert set(res["metric"]) == {"coverage-80", "coverage-95"}
    row95 = res[res["metric"] == "coverage-95"]
    assert np.isnan(row95["m1"].iloc[0])  # m1 has no 95% interval columns
    assert row95["m2"].iloc[0] == pytest.approx(1.0)


# --- v0.2.0 verifier regressions ------------------------------------------


def test_interval_metrics_drop_nan_bounds_consistently():
    """Rows with NaN interval bounds are excluded from every interval metric."""
    df = pd.DataFrame(
        {
            "unique_id": "s1",
            "ds": pd.date_range("2024-01-01", periods=4),
            "cutoff": pd.Timestamp("2023-12-31"),
            "y": [0.0, 0.0, 0.0, 10.0],
            "m": [0.0, 0.0, 0.0, 0.0],
            "m-lo-80": [-1.0, -1.0, -1.0, np.nan],
            "m-hi-80": [1.0, 1.0, 1.0, np.nan],
        }
    )
    out = evaluate(df, metrics=["coverage", "winkler"]).set_index("metric")["m"]
    # the NaN-bounds row (a miss if counted) is dropped -> full coverage
    assert out["coverage-80"] == 1.0
    assert out["winkler-80"] == 2.0  # width only, no violations

    all_nan = df.assign(**{"m-lo-80": np.nan, "m-hi-80": np.nan})
    out2 = evaluate(all_nan, metrics=["coverage", "winkler", "wis"]).set_index("metric")["m"]
    assert out2.isna().all()


# --- v0.10.0 audit regressions ---------------------------------------------


def test_evaluate_wis_scores_the_point_forecast():
    """The interval score used to ignore the point column entirely.

    ``_score_crps`` selected the model's point column into ``valid`` for the
    dropna but never scored it, so two models sharing lo/hi columns scored
    identically no matter how far apart their point forecasts were. The
    Weighted Interval Score carries a ``(1/2)|y - point|`` median term, so the
    worse point forecast must now score worse.
    """
    y = np.arange(8.0)
    df = pd.DataFrame(
        {
            "unique_id": "a",
            "ds": pd.date_range("2024-01-01", periods=8),
            "cutoff": pd.Timestamp("2023-12-31"),
            "y": y,
            "good": y,
            "awful": y + 97.5,
        }
    )
    for model in ("good", "awful"):
        df[f"{model}-lo-80"] = y - 1.2816
        df[f"{model}-hi-80"] = y + 1.2816
    res = evaluate(df, metrics=("wis",))
    assert res["awful"].iloc[0] > res["good"].iloc[0]


def test_evaluate_wis_is_the_weighted_interval_score():
    """Hand-computed WIS; the old uniform mean-of-pinball weighting is wrong.

    The previous formula averaged ``2 * pinball`` over the level-implied
    quantiles with uniform 1/K weights. Those quantiles are all tail, so the
    number came out at ~0.4-0.9x the true CRPS and moved non-monotonically
    with the requested level set. WIS weights interval k by ``alpha_k / 2``
    (the correct ``dq`` weight) and divides by ``K + 1/2``.
    """
    df = pd.DataFrame(
        {
            "unique_id": "a",
            "ds": pd.date_range("2024-01-01", periods=2),
            "cutoff": pd.Timestamp("2023-12-31"),
            "y": [0.0, 10.0],
            "m": [0.0, 0.0],
            "m-lo-80": [-1.0, -1.0],
            "m-hi-80": [1.0, 1.0],
        }
    )
    # alpha = 0.2; interval scores 2 and 2 + (2/0.2)*9 = 92 -> mean 47
    # 0.1 * 47 + 0.5 * mean(|y - 0|) = 4.7 + 2.5 = 7.2, over K + 1/2 = 1.5
    res = evaluate(df, metrics=("wis",))
    assert res["m"].iloc[0] == pytest.approx(4.8)


def _calibrated_normal_frame(levels):
    """N(0, 1) draws scored against the exact N(0, 1) predictive distribution.

    The true CRPS of this forecast is 1/sqrt(pi) = 0.5642, so the frame
    measures how far the emitted score sits from the CRPS at a given level set.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(7)
    y = rng.normal(0.0, 1.0, 40000)
    data = {"unique_id": "a", "ds": np.arange(y.size), "cutoff": 0, "y": y, "m": 0.0}
    for lvl in levels:
        a = 1.0 - lvl / 100.0
        data[f"m-lo-{lvl}"] = norm.ppf(a / 2)
        data[f"m-hi-{lvl}"] = norm.ppf(1 - a / 2)
    return pd.DataFrame(data)


@pytest.mark.parametrize(
    "levels, ratio",
    [
        ([80], 0.89),  # the ForecastEngine default level
        ([80, 95], 0.61),  # adding a level LOWERS the score: not monotone
        ([50, 80, 95], 0.76),
        (list(range(2, 99, 2)), 1.01),  # dense: quadrature is finally accurate
    ],
)
def test_wis_is_not_the_crps_on_the_level_sets_this_library_produces(levels, ratio):
    """Pins the documented ``wis / crps`` ratios, sparse level sets included.

    WIS is exact as WIS, but it is only a quadrature rule for the CRPS and the
    quadrature is poor on a handful of levels — 0.61x the true CRPS at
    ``[80, 95]``. The predecessor test only measured the 49-level regime, where
    the ratio is 1.01, and so validated the metric exactly where no default
    code path exercises it. These ratios are why the metric is emitted as
    ``wis`` rather than as ``crps``; if a future change makes it track the CRPS
    at sparse levels, this test fails and the name should be revisited.
    """
    true_crps = 1 / np.sqrt(np.pi)
    got = evaluate(_calibrated_normal_frame(levels), metrics=("wis",))["m"].iloc[0]
    assert got / true_crps == pytest.approx(ratio, abs=0.01)


# --- remediation round 2 ----------------------------------------------------


def test_the_emitted_interval_score_is_named_wis_not_crps():
    """The number is the Weighted Interval Score, so that is what it is called.

    Emitting it as ``crps`` mislabelled a value measured at 0.61x the true
    CRPS on ``level=[80, 95]``, which silently corrupts any comparison against
    properscoring/scoringrules — the whole point of a named, standard score.
    """
    res = evaluate(_degenerate_interval_frame(), metrics=("wis",))
    assert list(res["metric"]) == ["wis"]


def test_deprecated_crps_name_still_scores_and_warns():
    """v0.9.0's ``metrics=['crps']`` must keep working, under the honest label."""
    df = _degenerate_interval_frame()
    with pytest.warns(FutureWarning, match="renamed to 'wis'"):
        res = evaluate(df, metrics=("crps",))
    assert list(res["metric"]) == ["wis"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        pd.testing.assert_frame_equal(res, evaluate(df, metrics=("wis",)))


@pytest.mark.parametrize("metrics", [("crps", "wis"), ("wis", "crps")])
def test_alias_and_new_name_together_emit_one_row(metrics):
    """Asking for both spellings must not double the rows they share."""
    with pytest.warns(FutureWarning):
        res = evaluate(_degenerate_interval_frame(), metrics=metrics)
    assert list(res["metric"]) == ["wis"]


def test_coverage_excludes_nan_rows():
    """coverage() used to count NaN rows as interval misses.

    ``(y >= lo) & (y <= hi)`` is False for NaN, so an unknown actual or an
    unknown bound silently read as a miss and understated coverage. Rows with
    a non-finite y/lo/hi are now excluded, matching what evaluate() already
    does via ``_score_interval``'s dropna; an all-unusable input returns nan.
    """
    assert coverage([1, 2], [0, 0], [3, 3]) == 1.0
    assert coverage([1, np.nan], [0, 0], [3, 3]) == 1.0
    assert coverage([1, 2], [0, np.nan], [3, np.nan]) == 1.0
    assert np.isnan(coverage([np.nan, np.nan], [0, 0], [3, 3]))
    # a genuine miss still counts
    assert coverage([1, 9, np.nan], [0, 0, 0], [3, 3, 3]) == pytest.approx(0.5)


def test_pinball_loss_excludes_nan_rows():
    """pinball_loss() propagated NaN instead of excluding the unscoreable row.

    The module docstring promises the interval metrics (coverage, winkler,
    pinball, WIS) *exclude* rows whose inputs are NaN and return nan only when
    no row is scoreable. coverage()/winkler_score() do that via
    _scoreable_interval_rows, and evaluate()'s "pinball" gets it from
    _score_interval's dropna — but the exported pinball_loss took a plain
    np.mean, so one NaN actual or one NaN quantile forecast poisoned the whole
    score to nan. Drop the unscoreable rows and score the rest; scores for
    NaN-free input are unchanged.
    """
    # NaN-free input is unaffected
    assert pinball_loss([2, 1], [1, 2], q=0.9) == pytest.approx(0.5)
    # one unscoreable row is dropped, not propagated
    assert pinball_loss([2, np.nan], [1, 1], q=0.9) == pytest.approx(0.9)
    assert pinball_loss([2, 1], [1, np.nan], q=0.9) == pytest.approx(0.9)
    # nan only when nothing is scoreable
    assert np.isnan(pinball_loss([np.nan, np.nan], [1.0, 2.0], q=0.9))
    # dropping the row is equivalent to never having passed it
    assert pinball_loss([2, np.nan, 1], [1, 5.0, 2], q=0.9) == pytest.approx(
        pinball_loss([2, 1], [1, 2], q=0.9)
    )
    # ...and that really is what evaluate() computes for a NaN in y. Only the
    # NaN-in-y case is asserted: _score_interval drops a row when ANY of y, the
    # point forecast or either bound is NaN, so the two can legitimately differ
    # when a bound is the NaN. The docstring says so; this pins the case where
    # they must agree.
    frame = pd.DataFrame(
        {
            "unique_id": "a",
            "ds": pd.date_range("2020-01-01", periods=3),
            "y": [2.0, np.nan, 1.0],
            "M": [1.0, 5.0, 2.0],
            "M-lo-80": [0.0, 4.0, 1.0],
            "M-hi-80": [3.0, 6.0, 3.0],
        }
    )
    res = evaluate(frame, metrics=("pinball",))
    assert list(res["metric"]) == ["pinball-80"]
    # evaluate's pinball-80 averages both tails: q=0.1 against lo, q=0.9
    # against hi. Row 1 is dropped by both paths, so scoring the surviving two
    # rows directly must reproduce it exactly.
    expected = 0.5 * (
        pinball_loss([2.0, 1.0], [0.0, 1.0], q=0.1)
        + pinball_loss([2.0, 1.0], [3.0, 3.0], q=0.9)
    )
    assert res["M"].iloc[0] == pytest.approx(expected)


def test_winkler_score_excludes_nan_rows():
    """winkler_score() had the identical NaN-swallowing bug.

    ``np.where(y < lo, ...)`` is False for NaN, so a NaN row scored as a
    perfectly covered interval and quietly pulled the mean down.
    """
    assert winkler_score([1, np.nan], [0, 0], [3, 3], level=80) == pytest.approx(3.0)
    assert winkler_score([1, 2], [0, np.nan], [3, np.nan], level=80) == pytest.approx(3.0)
    assert np.isnan(winkler_score([np.nan], [0], [3], level=80))


@pytest.mark.parametrize("m", [0, -1, -3])
def test_scaled_metrics_reject_nonpositive_seasonality(m):
    """m <= 0 used to slip past the ``len(y_train) <= m`` guard.

    m=0 surfaced a raw numpy broadcast error; m<0 silently produced a
    meaningless scale (mase([6,7],[6.5,8],[1..5], m=-1) returned 0.1875). A
    seasonal period is >= 1 by definition, so reject it the way pinball_loss
    and winkler_score range-check their arguments.
    """
    with pytest.raises(ValueError, match="m must be a positive integer"):
        mase([6, 7], [6.5, 8], [1, 2, 3, 4, 5], m=m)
    with pytest.raises(ValueError, match="m must be a positive integer"):
        rmsse([6, 7], [6.5, 8], [1, 2, 3, 4, 5], m=m)
    with pytest.raises(ValueError, match="m must be a positive integer"):
        evaluate(_cv_frame(), metrics=("mase",), train_df=_train_frame(), seasonality=m)


def test_evaluate_missing_train_series_names_the_series():
    """A series in cv_df but absent from train_df blamed the seasonality arg.

    The empty training slice fell through to _naive_scale, which raised
    'training series (len 0) must be longer than m=1' — a message naming m and
    never naming the series or train_df. Diagnose it at the call site instead.
    """
    cv = _cv_frame()
    train = _train_frame()
    train = train[train["unique_id"] != "a"]
    with pytest.raises(ValueError, match=r"train_df has no rows for series 'a'"):
        evaluate(cv, metrics=("mase",), train_df=train)
