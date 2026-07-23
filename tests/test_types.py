"""Tests for the panel data contract (core.types)."""

import numpy as np
import pandas as pd
import pytest

from forecast_os.core.exceptions import DataContractError
from forecast_os.core.types import future_ds, infer_step, to_panel, validate_panel


def _panel(uid="a", n=5):
    return pd.DataFrame(
        {
            "unique_id": uid,
            "ds": pd.date_range("2024-01-01", periods=n, freq="D"),
            "y": np.arange(n, dtype=float),
        }
    )


# -- validate_panel: rejections ---------------------------------------------


def test_non_dataframe_raises():
    with pytest.raises(DataContractError, match="pandas DataFrame"):
        validate_panel([1, 2, 3])


def test_missing_column_raises():
    df = _panel().drop(columns=["y"])
    with pytest.raises(DataContractError, match=r"missing required column.*'y'"):
        validate_panel(df)


def test_missing_multiple_columns_message_lists_them():
    df = pd.DataFrame({"y": [1.0]})
    with pytest.raises(DataContractError, match=r"'unique_id'.*'ds'"):
        validate_panel(df)


def test_empty_panel_raises():
    df = _panel().iloc[:0]
    with pytest.raises(DataContractError, match="empty"):
        validate_panel(df)


def test_duplicate_pair_raises():
    df = pd.concat([_panel(), _panel().iloc[[2]]], ignore_index=True)
    with pytest.raises(DataContractError, match="duplicate"):
        validate_panel(df)


def test_nan_y_raises_with_hint():
    df = _panel()
    df.loc[2, "y"] = np.nan
    with pytest.raises(DataContractError, match="NaN"):
        validate_panel(df)


def test_non_numeric_y_raises():
    df = _panel()
    df["y"] = ["a", "b", "c", "d", "e"]
    with pytest.raises(DataContractError, match="numeric"):
        validate_panel(df)


@pytest.mark.filterwarnings("ignore::UserWarning")  # pandas format-inference chatter
def test_unparseable_ds_raises():
    df = _panel()
    df["ds"] = ["not-a-date"] * 5
    with pytest.raises(DataContractError, match="ds"):
        validate_panel(df)


# -- validate_panel: normalization ------------------------------------------


def test_allow_missing_passes_nan():
    df = _panel()
    df.loc[2, "y"] = np.nan
    out = validate_panel(df, allow_missing=True)
    assert out["y"].isna().sum() == 1


def test_unsorted_input_comes_back_sorted():
    df = pd.concat([_panel("b"), _panel("a")], ignore_index=True)
    df = df.sample(frac=1.0, random_state=0)
    out = validate_panel(df)
    assert list(out["unique_id"].unique()) == ["a", "b"]
    for _, g in out.groupby("unique_id"):
        assert g["ds"].is_monotonic_increasing
    # original untouched (copy semantics)
    assert not df.equals(out)


def test_string_ds_parsed_to_datetime():
    df = _panel()
    df["ds"] = df["ds"].dt.strftime("%Y-%m-%d")
    out = validate_panel(df)
    assert pd.api.types.is_datetime64_any_dtype(out["ds"])
    assert out["ds"].iloc[0] == pd.Timestamp("2024-01-01")


def test_integer_ds_preserved():
    df = _panel()
    df["ds"] = np.arange(5)
    out = validate_panel(df)
    assert pd.api.types.is_numeric_dtype(out["ds"])
    assert not pd.api.types.is_datetime64_any_dtype(out["ds"])


def test_y_coerced_to_float():
    df = _panel()
    df["y"] = np.arange(5)  # ints
    out = validate_panel(df)
    assert out["y"].dtype == float


# -- to_panel ----------------------------------------------------------------


def test_to_panel_defaults():
    out = to_panel([1, 2, 3])
    assert list(out.columns) == ["unique_id", "ds", "y"]
    assert (out["unique_id"] == "series-0").all()
    assert list(out["ds"]) == [0, 1, 2]
    assert out["y"].dtype == float
    validate_panel(out)  # satisfies the contract


def test_to_panel_explicit_ds_and_id():
    ds = pd.date_range("2024-01-01", periods=3, freq="D")
    out = to_panel([1.0, 2.0, 3.0], ds=ds, unique_id="x")
    assert (out["unique_id"] == "x").all()
    assert list(out["ds"]) == list(ds)


# -- infer_step ---------------------------------------------------------------


def test_infer_step_daily():
    ds = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D"))
    assert infer_step(ds) == "D"


def test_infer_step_monthly():
    ds = pd.Series(pd.date_range("2024-01-01", periods=12, freq="MS"))
    assert infer_step(ds) == "MS"


def test_infer_step_irregular_datetime_median():
    ds = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-08"]))
    assert infer_step(ds) == pd.Timedelta(days=2)


def test_infer_step_two_datetimes():
    ds = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-04"]))
    assert infer_step(ds) == pd.Timedelta(days=3)


def test_infer_step_integer():
    step = infer_step(pd.Series([0, 2, 4, 6]))
    assert step == 2
    assert isinstance(step, int)


def test_infer_step_float():
    assert infer_step(pd.Series([0.0, 0.5, 1.0])) == 0.5


def test_infer_step_single_value_defaults_to_one():
    assert infer_step(pd.Series([5])) == 1


# -- future_ds ----------------------------------------------------------------


def test_future_ds_datetime_freq_string():
    out = future_ds(pd.Timestamp("2024-01-05"), 3, "D")
    assert list(out) == list(pd.to_datetime(["2024-01-06", "2024-01-07", "2024-01-08"]))


def test_future_ds_datetime_timedelta_step():
    out = future_ds(pd.Timestamp("2024-01-01"), 2, pd.Timedelta(days=2))
    assert list(out) == list(pd.to_datetime(["2024-01-03", "2024-01-05"]))


def test_future_ds_monthly():
    out = future_ds(pd.Timestamp("2024-03-01"), 2, "MS")
    assert list(out) == list(pd.to_datetime(["2024-04-01", "2024-05-01"]))


def test_future_ds_integer():
    out = future_ds(10, 3, 2)
    np.testing.assert_array_equal(out, [12, 14, 16])


def test_future_ds_continues_after_last():
    """Every generated stamp is strictly after last_ds, strictly increasing."""
    for last, step in [(pd.Timestamp("2024-06-30"), "D"), (7, 1)]:
        out = pd.Series(list(future_ds(last, 5, step)))
        assert (out > last).all()
        assert out.is_monotonic_increasing
        assert len(out) == 5
