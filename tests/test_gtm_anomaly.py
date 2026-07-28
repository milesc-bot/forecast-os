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


class TestNullSegmentKeysAreScanned:
    """Regression for silently unmonitored null-keyed segments (v0.9.0 audit SHOULD-5).

    ``df.groupby(by)`` inherited pandas' default ``dropna=True``, so every row
    whose segment label was null was dropped before scanning: an unassigned
    owner/team/region — an everyday CRM export state — came back as an ordinary
    empty result rather than as an alert, and with the default ``by='unique_id'``
    a panel with null ids returned empty outright. The internal inconsistency was
    stark: the same function hard-errors on a NaN in the VALUE column but
    discarded NaN segment keys without a word. Null keys are now their own
    segment.
    """

    def _panel(self, n=10):
        ds = list(pd.date_range("2026-01-01", periods=n, freq="MS"))
        return pd.DataFrame(
            {
                "team": ["east"] * n + [None] * n,
                TIME_COL: ds * 2,
                TARGET_COL: [10.0] * n + [10.0] * (n - 1) + [1000.0],
            }
        )

    def test_spike_in_the_null_segment_is_flagged(self):
        out = detect_anomalies(self._panel(), by="team", window=6, threshold=3.0)
        assert len(out) == 1
        row = out.iloc[0]
        assert pd.isna(row["team"])
        assert row["value"] == 1000.0
        assert row["severity"] == "critical"

    def test_null_segment_matches_scanning_that_segment_alone(self):
        panel = self._panel()
        alone = panel[panel["team"].isna()].drop(columns=["team"])
        together = detect_anomalies(panel, by="team", window=6, threshold=3.0)
        separate = detect_anomalies(alone, by=None, window=6, threshold=3.0)
        assert len(together) == len(separate) == 1
        assert together["score"].iloc[0] == separate["score"].iloc[0]

    def test_null_unique_id_panel_is_not_silently_empty(self):
        n = 10
        panel = pd.DataFrame(
            {
                ID_COL: [None] * n,
                TIME_COL: pd.date_range("2026-01-01", periods=n, freq="MS"),
                TARGET_COL: [10.0] * (n - 1) + [1000.0],
            }
        )
        out = detect_anomalies(panel, window=6, threshold=3.0)
        assert len(out) == 1

    def test_labelled_segments_are_unaffected(self):
        panel = self._panel()
        panel.loc[panel["team"].isna(), "team"] = "west"
        out = detect_anomalies(panel, by="team", window=6, threshold=3.0)
        assert list(out["team"]) == ["west"]


class TestEmptyResultDtypes:
    """Regression for the object-dtype 'ds' on the empty path (v0.9.0 audit NIT-7).

    The empty-frame builder typed only ``value``/``expected``/``score`` as
    float64 and defaulted everything else to object, so a segment that flagged
    nothing produced an object-dtype ``ds``. ``pd.concat``-ing that with a
    flagged segment's datetime64 ``ds`` downgraded the whole column to object and
    ``c['ds'].dt.year`` then raised AttributeError. The empty frame now takes its
    ``ds`` dtype from the input, which also keeps an integer-age panel integer.
    """

    def _clean(self):
        return pd.DataFrame(
            {
                ID_COL: ["a"] * 10,
                TIME_COL: pd.date_range("2026-01-01", periods=10, freq="MS"),
                TARGET_COL: [1.0] * 10,
            }
        )

    def _flagged(self):
        return pd.DataFrame(
            {
                ID_COL: ["b"] * 8,
                TIME_COL: pd.date_range("2026-01-01", periods=8, freq="MS"),
                TARGET_COL: [10.0] * 7 + [1000.0],
            }
        )

    def test_empty_and_flagged_results_share_the_ds_dtype(self):
        empty = detect_anomalies(self._clean(), window=6)
        flagged = detect_anomalies(self._flagged(), window=6)
        assert len(empty) == 0 and len(flagged) == 1
        assert empty[TIME_COL].dtype == flagged[TIME_COL].dtype

    def test_concat_of_empty_and_flagged_keeps_the_dt_accessor(self):
        empty = detect_anomalies(self._clean(), window=6)
        flagged = detect_anomalies(self._flagged(), window=6)
        both = pd.concat([empty, flagged], ignore_index=True)
        assert both[TIME_COL].dt.year.tolist() == [2026]

    def test_integer_age_panel_keeps_an_integer_ds(self):
        ages = pd.DataFrame(
            {ID_COL: ["c"] * 10, TIME_COL: np.arange(10), TARGET_COL: [1.0] * 10}
        )
        out = detect_anomalies(ages, window=6)
        assert len(out) == 0
        assert out[TIME_COL].dtype == ages[TIME_COL].dtype

    def test_computed_columns_keep_float64(self):
        out = detect_anomalies(self._clean(), window=6)
        for col in ("value", "expected", "score"):
            assert out[col].dtype == np.dtype("float64")


class TestIQRZeroScaleIsDocumentedNotAccidental:
    """Pins the zero-scale rule the module docstring now states honestly
    (v0.9.0 audit NIT-9).

    The docstring justified the ``inf``/critical branch as "the trailing window
    is perfectly flat", which is true for ``zscore`` but not for ``iqr``: ``iqr``
    only needs a flat interquartile CORE, so a window with one wild point still
    has ``IQR == 0`` and any later move — however small — scores ``inf``. The
    control below shows the wild point is irrelevant to the iqr flag (a flat
    window flags under BOTH methods); its only effect is to silence ``zscore`` by
    inflating the std. Behaviour is unchanged; the docstring is what was wrong.
    """

    def _series(self, vals):
        return pd.DataFrame(
            {
                ID_COL: ["a"] * len(vals),
                TIME_COL: pd.date_range("2026-01-01", periods=len(vals), freq="MS"),
                TARGET_COL: [float(v) for v in vals],
            }
        )

    def test_flat_window_flags_under_both_methods(self):
        df = self._series([10, 10, 10, 10, 10, 10, 11])
        for method in ("iqr", "zscore"):
            out = detect_anomalies(df, method=method, window=6, threshold=3.0)
            assert len(out) == 1, method
            assert out["score"].iloc[0] == np.inf
            assert out["severity"].iloc[0] == "critical"

    def test_flat_core_with_one_wild_point_flags_iqr_only(self):
        df = self._series([10, 10, 10, 10, 10, 1000, 11])
        iqr = detect_anomalies(df, method="iqr", window=6, threshold=3.0)
        assert len(iqr) == 1 and iqr["score"].iloc[0] == np.inf
        # zscore's std is inflated by the wild point, so it says nothing
        assert len(detect_anomalies(df, method="zscore", window=6, threshold=3.0)) == 0
