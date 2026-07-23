"""Hand-computed tests for evaluation.metrics."""

import numpy as np
import pandas as pd
import pytest

from forecast_os.evaluation.metrics import (
    coverage,
    evaluate,
    mae,
    mape,
    mase,
    pinball_loss,
    rmse,
    rmsse,
    smape,
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
