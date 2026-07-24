"""Tests for the pure-pandas snapshot analysis helpers (snapshots.analysis)."""

import numpy as np
import pandas as pd
import pytest

from forecast_os.snapshots import (
    accuracy_over_time,
    forecast_vs_actual,
    snapshot_evolution,
)


class TestSnapshotEvolution:
    def test_picks_fixed_target_period_across_snapshots(self):
        # two as_of snapshots each recording periods 2024-04 and 2024-05 for
        # series a; we ask "how did our 2024-04 number move week over week".
        hist = pd.DataFrame(
            {
                "unique_id": ["a", "a", "a", "a"],
                "ds": pd.to_datetime(
                    ["2024-04-01", "2024-05-01", "2024-04-01", "2024-05-01"]
                ),
                "y": [100.0, 200.0, 150.0, 250.0],
                "as_of": pd.to_datetime(
                    ["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"]
                ),
            }
        )
        ev = snapshot_evolution(hist, target_ds="2024-04-01")
        assert list(ev.columns) == ["unique_id", "as_of", "value"]
        assert list(ev["as_of"]) == [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-02-01"),
        ]
        assert list(ev["value"]) == [100.0, 150.0]

    def test_works_for_forecast_history_and_series_filter(self):
        fh = pd.DataFrame(
            {
                "unique_id": ["a", "b", "a", "b"],
                "ds": pd.to_datetime(
                    ["2024-04-01", "2024-04-01", "2024-04-01", "2024-04-01"]
                ),
                "yhat": [90.0, 500.0, 140.0, 510.0],
                "as_of": pd.to_datetime(
                    ["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"]
                ),
            }
        )
        ev = snapshot_evolution(fh, target_ds="2024-04-01", value_col="yhat", series="a")
        assert set(ev["unique_id"]) == {"a"}
        assert list(ev["value"]) == [90.0, 140.0]


class TestForecastVsActual:
    def test_joins_actuals_drops_future_ds_and_nan_safe_pct(self):
        # forecast for 2024-03 has no actual yet -> dropped; 2024-02 actual is
        # zero -> pct_error must be nan (nan-safe), error still computed.
        fh = pd.DataFrame(
            {
                "unique_id": ["a", "a", "a"],
                "ds": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
                "yhat": [10.0, 5.0, 30.0],
                "as_of": pd.to_datetime(["2023-12-01", "2023-12-01", "2023-12-01"]),
            }
        )
        actual = pd.DataFrame(
            {
                "unique_id": ["a", "a"],
                "ds": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                "y": [8.0, 0.0],
            }
        )
        fva = forecast_vs_actual(fh, actual)
        assert list(fva.columns) == [
            "unique_id",
            "as_of",
            "ds",
            "forecast",
            "actual",
            "error",
            "abs_error",
            "pct_error",
        ]
        # the unmatched future ds is dropped
        assert len(fva) == 2
        assert pd.Timestamp("2024-03-01") not in set(fva["ds"])

        r1 = fva[fva["ds"] == pd.Timestamp("2024-01-01")].iloc[0]
        assert r1["forecast"] == 10.0
        assert r1["actual"] == 8.0
        assert r1["error"] == 2.0
        assert r1["abs_error"] == 2.0
        assert r1["pct_error"] == pytest.approx(2.0 / 8.0)

        r2 = fva[fva["ds"] == pd.Timestamp("2024-02-01")].iloc[0]
        assert r2["error"] == 5.0
        assert r2["abs_error"] == 5.0
        assert np.isnan(r2["pct_error"])


class TestAccuracyOverTime:
    def test_aggregates_per_as_of_with_signed_bias(self):
        # as_of 2024-01: forecasts run high (bias +3); as_of 2024-02: run low (-3)
        frame = pd.DataFrame(
            {
                "unique_id": ["a", "b", "a", "b"],
                "as_of": pd.to_datetime(
                    ["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"]
                ),
                "ds": pd.to_datetime(
                    ["2024-03-01", "2024-03-01", "2024-03-01", "2024-03-01"]
                ),
                "forecast": [12.0, 14.0, 8.0, 6.0],
                "actual": [10.0, 10.0, 10.0, 10.0],
                "error": [2.0, 4.0, -2.0, -4.0],
                "abs_error": [2.0, 4.0, 2.0, 4.0],
                "pct_error": [0.2, 0.4, -0.2, -0.4],
            }
        )
        acc = accuracy_over_time(frame)
        assert list(acc.columns) == ["as_of", "n", "mae", "bias", "pct_bias"]
        assert list(acc["as_of"]) == [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-02-01"),
        ]

        g1 = acc.iloc[0]
        assert g1["n"] == 2
        assert g1["mae"] == 3.0
        assert g1["bias"] == 3.0  # runs high
        assert g1["pct_bias"] == pytest.approx(6.0 / 20.0)

        g2 = acc.iloc[1]
        assert g2["bias"] == -3.0  # runs low
        assert g2["pct_bias"] == pytest.approx(-6.0 / 20.0)
