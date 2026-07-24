"""Tests for funnel/time-series anomaly detection (gtm.anomaly)."""

import numpy as np
import pandas as pd
import pytest

from forecast_os.core.exceptions import DataContractError
from forecast_os.core.types import ID_COL, TARGET_COL, TIME_COL
from forecast_os.gtm.anomaly import detect_anomalies

_OUT_COLS = [ID_COL, TIME_COL, "value", "expected", "score", "severity"]


def _alt_series(unique_id, n=30, lo=100.0, hi=101.0):
    """Alternating lo/hi baseline: nonzero but tiny trailing std."""
    y = np.array([lo if i % 2 == 0 else hi for i in range(n)], dtype=float)
    ds = pd.date_range("2022-01-01", periods=n, freq="MS")
    return pd.DataFrame({ID_COL: unique_id, TIME_COL: ds, TARGET_COL: y}), y, ds


class TestZScore:
    def test_planted_spike_flagged_with_positive_sign(self):
        df, y, ds = _alt_series("s1")
        df.loc[20, TARGET_COL] = 120.0  # sharp spike well above trailing history
        out = detect_anomalies(df, window=6, threshold=3.0)
        assert len(out) == 1
        row = out.iloc[0]
        assert row[TIME_COL] == ds[20]
        assert row["value"] == 120.0
        assert row["expected"] == pytest.approx(100.5)
        assert row["score"] > 0  # spike -> positive z
        assert row["severity"] == "critical"

    def test_planted_drop_flagged_with_negative_sign(self):
        df, y, ds = _alt_series("s1")
        df.loc[20, TARGET_COL] = 80.0  # sharp drop below trailing history
        out = detect_anomalies(df, window=6, threshold=3.0)
        assert len(out) == 1
        assert out.iloc[0][TIME_COL] == ds[20]
        assert out.iloc[0]["score"] < 0  # drop -> negative z

    def test_clean_series_flags_nothing(self):
        df, y, ds = _alt_series("s1")
        out = detect_anomalies(df, window=6, threshold=3.0)
        assert len(out) == 0
        assert list(out.columns) == _OUT_COLS

    def test_output_columns_present_when_flagged(self):
        df, y, ds = _alt_series("s1")
        df.loc[20, TARGET_COL] = 120.0
        out = detect_anomalies(df, window=6, threshold=3.0)
        assert list(out.columns) == _OUT_COLS

    def test_unsorted_input_flags_correct_period(self):
        df, y, ds = _alt_series("s1")
        df.loc[20, TARGET_COL] = 120.0
        shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
        out = detect_anomalies(shuffled, window=6, threshold=3.0)
        assert len(out) == 1
        assert out.iloc[0][TIME_COL] == ds[20]


class TestThresholdSensitivity:
    def _series_z4(self):
        # alternating 100/102 -> trailing mean 101, std 1.0; a 105 point is z=4.
        df, y, ds = _alt_series("s1", lo=100.0, hi=102.0)
        df.loc[20, TARGET_COL] = 105.0
        return df, ds

    def test_low_threshold_flags(self):
        df, ds = self._series_z4()
        out = detect_anomalies(df, window=6, threshold=3.0)
        assert len(out) == 1
        assert out.iloc[0][TIME_COL] == ds[20]
        assert out.iloc[0]["score"] == pytest.approx(4.0)

    def test_high_threshold_suppresses(self):
        df, ds = self._series_z4()
        out = detect_anomalies(df, window=6, threshold=5.0)
        assert len(out) == 0


class TestSegmentedPanel:
    def test_only_anomalous_segment_flagged(self):
        amer, _, ds = _alt_series("AMER")
        emea, _, _ = _alt_series("EMEA")
        emea.loc[20, TARGET_COL] = 70.0  # EMEA conversion dropped
        panel = pd.concat([amer, emea], ignore_index=True)
        out = detect_anomalies(panel, window=6, threshold=3.0)
        assert set(out[ID_COL]) == {"EMEA"}
        assert len(out) == 1
        assert out.iloc[0][TIME_COL] == ds[20]
        assert out.iloc[0]["score"] < 0


class TestZeroVarianceHistory:
    def test_flat_then_drop_flagged(self):
        n = 30
        y = np.full(n, 100.0)
        y[20] = 50.0  # drop from a perfectly flat baseline
        ds = pd.date_range("2022-01-01", periods=n, freq="MS")
        df = pd.DataFrame({ID_COL: "s1", TIME_COL: ds, TARGET_COL: y})
        out = detect_anomalies(df, window=6, threshold=3.0)
        assert len(out) == 1
        assert out.iloc[0][TIME_COL] == ds[20]
        assert out.iloc[0]["score"] < 0
        assert out.iloc[0]["severity"] == "critical"

    def test_perfectly_flat_series_flags_nothing(self):
        n = 30
        ds = pd.date_range("2022-01-01", periods=n, freq="MS")
        df = pd.DataFrame({ID_COL: "s1", TIME_COL: ds, TARGET_COL: np.full(n, 100.0)})
        out = detect_anomalies(df, window=6, threshold=3.0)
        assert len(out) == 0


class TestIQRMethod:
    def test_iqr_flags_spike_with_median_expected(self):
        df, y, ds = _alt_series("s1", lo=100.0, hi=102.0)
        df.loc[20, TARGET_COL] = 120.0
        out = detect_anomalies(df, method="iqr", window=6, threshold=3.0)
        assert len(out) == 1
        row = out.iloc[0]
        assert row[TIME_COL] == ds[20]
        assert row["expected"] == pytest.approx(101.0)  # rolling median
        assert row["score"] > 0


class TestUngroupedSingleSeries:
    def test_by_none_single_series(self):
        df, y, ds = _alt_series("s1")
        df.loc[20, TARGET_COL] = 120.0
        df = df.drop(columns=[ID_COL])
        out = detect_anomalies(df, by=None, window=6, threshold=3.0)
        assert list(out.columns) == [TIME_COL, "value", "expected", "score", "severity"]
        assert len(out) == 1
        assert out.iloc[0][TIME_COL] == ds[20]


class TestValidation:
    def test_missing_value_column_raises(self):
        df, y, ds = _alt_series("s1")
        with pytest.raises(DataContractError, match="revenue"):
            detect_anomalies(df, value_col="revenue")

    def test_missing_by_column_raises(self):
        df, y, ds = _alt_series("s1")
        with pytest.raises(DataContractError, match="segment"):
            detect_anomalies(df, by="segment")

    def test_missing_ds_column_raises(self):
        df, y, ds = _alt_series("s1")
        with pytest.raises(DataContractError, match="ds"):
            detect_anomalies(df.drop(columns=[TIME_COL]))

    def test_unknown_method_raises(self):
        df, y, ds = _alt_series("s1")
        with pytest.raises(ValueError, match="method"):
            detect_anomalies(df, method="bogus")

    def test_window_too_small_raises(self):
        df, y, ds = _alt_series("s1")
        with pytest.raises(ValueError, match="window"):
            detect_anomalies(df, window=1)

    def test_non_positive_threshold_raises(self):
        df, y, ds = _alt_series("s1")
        with pytest.raises(ValueError, match="threshold"):
            detect_anomalies(df, threshold=0.0)

    def test_nan_value_raises(self):
        df, y, ds = _alt_series("s1")
        df.loc[5, TARGET_COL] = np.nan
        with pytest.raises(DataContractError, match="NaN"):
            detect_anomalies(df)

    def test_non_dataframe_raises(self):
        with pytest.raises(DataContractError, match="DataFrame"):
            detect_anomalies([1, 2, 3])
