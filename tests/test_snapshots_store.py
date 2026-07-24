"""Tests for the snapshot store (snapshots.store.SnapshotStore)."""

import pandas as pd
import pytest

from forecast_os.core.exceptions import DataContractError, ForecastOSError
from forecast_os.core.types import validate_panel
from forecast_os.snapshots import SnapshotStore
from forecast_os.snapshots import store as store_module


def _panel(values):
    """A tiny 2-series monthly panel (a, b) with 2 periods each."""
    return pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01"]),
            "y": [float(v) for v in values],
        }
    )


def _forecast(yhat):
    """A tiny committed forecast frame (unique_id, ds, yhat)."""
    return pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": pd.to_datetime(["2024-03-01", "2024-04-01"]),
            "yhat": [float(v) for v in yhat],
        }
    )


class TestRoundTrip:
    def test_snapshot_load_history_across_two_as_of_and_both_kinds(self, tmp_path):
        store = SnapshotStore(tmp_path)
        p1 = _panel([1, 2, 3, 4])
        e1 = store.snapshot(p1, as_of="2024-02-15")
        p2 = _panel([10, 20, 30, 40])
        store.snapshot(p2, as_of="2024-03-15", label="week2")
        f1 = _forecast([5, 6])
        store.snapshot(f1, as_of="2024-02-15", kind="forecast")

        # snapshot() returns a manifest entry dict
        assert isinstance(e1, dict)
        assert e1["kind"] == "panel"
        assert e1["rows"] == 4
        assert e1["file"].endswith("20240215.parquet")

        # as_of_dates are per-kind, unique, sorted
        assert store.as_of_dates() == [
            pd.Timestamp("2024-02-15"),
            pd.Timestamp("2024-03-15"),
        ]
        assert store.as_of_dates(kind="forecast") == [pd.Timestamp("2024-02-15")]

        # load a specific as_of returns the validated panel round-tripped
        loaded = store.load(as_of="2024-02-15")
        pd.testing.assert_frame_equal(loaded, validate_panel(p1), check_dtype=False)

        # forecast round-trips too
        loaded_fc = store.load(as_of="2024-02-15", kind="forecast")
        assert set(loaded_fc["yhat"]) == {5.0, 6.0}

        # load(None) is the most recent as_of overall
        latest = store.load()
        assert set(latest["y"]) == {10.0, 20.0, 30.0, 40.0}

        # history stacks every panel snapshot with an as_of column
        hist = store.history()
        assert "as_of" in hist.columns
        assert len(hist) == 8
        assert set(pd.to_datetime(hist["as_of"].unique())) == {
            pd.Timestamp("2024-02-15"),
            pd.Timestamp("2024-03-15"),
        }
        # sorted by (as_of, unique_id, ds)
        assert list(hist["as_of"]) == sorted(hist["as_of"])

    def test_history_series_filter(self, tmp_path):
        store = SnapshotStore(tmp_path)
        store.snapshot(_panel([1, 2, 3, 4]), as_of="2024-02-15")
        only_a = store.history(series="a")
        assert set(only_a["unique_id"]) == {"a"}
        both = store.history(series=["a", "b"])
        assert set(both["unique_id"]) == {"a", "b"}

    def test_manifest_dataframe(self, tmp_path):
        store = SnapshotStore(tmp_path)
        store.snapshot(_panel([1, 2, 3, 4]), as_of="2024-02-15", label="w1")
        store.snapshot(_forecast([5, 6]), as_of="2024-02-15", kind="forecast")
        m = store.manifest()
        assert list(m.columns) == ["as_of", "kind", "label", "file", "rows", "created_at"]
        assert len(m) == 2
        assert set(m["kind"]) == {"panel", "forecast"}
        assert pd.api.types.is_datetime64_any_dtype(m["as_of"])


class TestAppendSuffixing:
    def test_duplicate_as_of_kind_gets_numeric_suffix_and_keeps_all(self, tmp_path):
        store = SnapshotStore(tmp_path)
        store.snapshot(_panel([1, 1, 1, 1]), as_of="2024-02-15")
        store.snapshot(_panel([9, 9, 9, 9]), as_of="2024-02-15")

        files = sorted(p.name for p in (tmp_path / "panel").glob("*.parquet"))
        assert files == ["20240215-1.parquet", "20240215.parquet"]

        # manifest keeps both
        m = store.manifest()
        assert (m["kind"] == "panel").sum() == 2

        # history stacks both snapshots of the same date
        hist = store.history()
        assert len(hist) == 8

    def test_load_returns_latest_snapshot_for_a_date(self, tmp_path):
        store = SnapshotStore(tmp_path)
        store.snapshot(_panel([1, 1, 1, 1]), as_of="2024-02-15")
        store.snapshot(_panel([9, 9, 9, 9]), as_of="2024-02-15")
        loaded = store.load(as_of="2024-02-15")
        assert set(loaded["y"]) == {9.0}


class TestValidation:
    def test_panel_kind_rejects_bad_panel(self, tmp_path):
        store = SnapshotStore(tmp_path)
        bad = pd.DataFrame(
            {"unique_id": ["a"], "ds": ["2024-01-01"], "yhat": [1.0]}  # missing y
        )
        with pytest.raises(DataContractError):
            store.snapshot(bad, as_of="2024-01-01", kind="panel")

    def test_forecast_kind_requires_yhat(self, tmp_path):
        store = SnapshotStore(tmp_path)
        bad = pd.DataFrame(
            {"unique_id": ["a"], "ds": ["2024-01-01"], "y": [1.0]}  # missing yhat
        )
        with pytest.raises(DataContractError):
            store.snapshot(bad, as_of="2024-01-01", kind="forecast")

    def test_unknown_kind_raises(self, tmp_path):
        store = SnapshotStore(tmp_path)
        with pytest.raises(ForecastOSError):
            store.snapshot(_panel([1, 2, 3, 4]), as_of="2024-01-01", kind="bogus")


class TestLoadErrors:
    def test_load_empty_store_raises(self, tmp_path):
        store = SnapshotStore(tmp_path)
        with pytest.raises(ForecastOSError):
            store.load()

    def test_load_unknown_as_of_raises(self, tmp_path):
        store = SnapshotStore(tmp_path)
        store.snapshot(_panel([1, 2, 3, 4]), as_of="2024-02-15")
        with pytest.raises(ForecastOSError):
            store.load(as_of="2025-01-01")


class TestCreatedAt:
    def test_created_at_defaults_to_as_of_and_is_overridable(self, tmp_path):
        store = SnapshotStore(tmp_path)
        e = store.snapshot(_panel([1, 2, 3, 4]), as_of="2024-02-15")
        assert e["created_at"] == pd.Timestamp("2024-02-15").isoformat()
        e2 = store.snapshot(
            _panel([1, 2, 3, 4]), as_of="2024-03-15", created_at="2024-03-16T09:30:00"
        )
        assert e2["created_at"] == "2024-03-16T09:30:00"


class TestImportGuard:
    def test_construction_without_pyarrow_raises_extras_hint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(store_module, "_HAS_PYARROW", False)
        with pytest.raises(ImportError, match=r"forecast-os\[snapshots\]"):
            SnapshotStore(tmp_path)


def test_corrupt_manifest_raises_forecast_os_error(tmp_path):
    from forecast_os.core.exceptions import ForecastOSError
    from forecast_os.snapshots.store import SnapshotStore

    store = SnapshotStore(tmp_path)
    (tmp_path / "manifest.json").write_text("{not valid json")
    with pytest.raises(ForecastOSError, match="corrupt snapshot manifest"):
        store.load()


def test_second_resolution_ds_round_trips_dtype_stable(tmp_path):
    from forecast_os.snapshots.store import SnapshotStore

    ds = pd.DatetimeIndex(["2026-01-05", "2026-01-12", "2026-01-19"]).as_unit("s")
    panel = pd.DataFrame({"unique_id": "s", "ds": ds, "y": [1.0, 2.0, 3.0]})
    store = SnapshotStore(tmp_path)
    store.snapshot(panel, as_of="2026-01-19", kind="panel")
    loaded = store.load(kind="panel")
    assert loaded["ds"].dtype == panel["ds"].dtype  # [s] preserved, not promoted to [ms]
    pd.testing.assert_frame_equal(loaded, panel)
