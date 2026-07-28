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


class TestTimezoneAwareDs:
    """A tz-aware ``ds`` must survive the write/read round trip.

    ``validate_panel`` accepts a tz-aware ``ds`` and ``snapshot()`` wrote the
    parquet file happily, but ``_restore_ds_unit`` re-applied the recorded
    resolution with ``astype("datetime64[<unit>]")`` — a tz-aware -> tz-naive
    cast pandas refuses — so ``load()``/``history()`` raised ``TypeError`` and
    the data the store had accepted was permanently unreadable through every
    store API. One tz-aware snapshot also broke ``history()`` for the whole
    kind. The restore is now unit-only (``dt.as_unit``), which is tz-preserving:
    the resolution is still pinned, and the timezone is carried through
    untouched rather than being dropped or silently converted to UTC.
    """

    @staticmethod
    def _tz_panel(tz):
        ds = pd.to_datetime(["2024-01-01", "2024-02-01"]).tz_localize(tz)
        return pd.DataFrame({"unique_id": ["a", "a"], "ds": ds, "y": [1.0, 2.0]})

    @pytest.mark.parametrize("tz", ["UTC", "US/Eastern"])
    def test_tz_aware_panel_round_trips_through_load(self, tmp_path, tz):
        store = SnapshotStore(tmp_path)
        panel = self._tz_panel(tz)
        store.snapshot(panel, as_of="2024-02-15", kind="panel")

        loaded = store.load(kind="panel")
        assert loaded["ds"].dtype == panel["ds"].dtype  # unit AND tz preserved
        pd.testing.assert_frame_equal(loaded, validate_panel(panel))

    def test_tz_aware_forecast_round_trips_through_history(self, tmp_path):
        store = SnapshotStore(tmp_path)
        ds = pd.to_datetime(["2024-03-01", "2024-04-01"]).tz_localize("UTC")
        forecast = pd.DataFrame({"unique_id": ["a", "a"], "ds": ds, "yhat": [5.0, 6.0]})
        store.snapshot(forecast, as_of="2024-02-15", kind="forecast")

        hist = store.history(kind="forecast")
        assert len(hist) == 2
        assert hist["ds"].dtype == forecast["ds"].dtype
        assert list(hist["ds"]) == list(ds)

    def test_tz_aware_second_resolution_keeps_both_unit_and_tz(self, tmp_path):
        store = SnapshotStore(tmp_path)
        ds = pd.DatetimeIndex(["2026-01-05", "2026-01-12"]).as_unit("s")
        ds = ds.tz_localize("US/Eastern")
        panel = pd.DataFrame({"unique_id": "s", "ds": ds, "y": [1.0, 2.0]})
        store.snapshot(panel, as_of="2026-01-12", kind="panel")

        loaded = store.load(kind="panel")
        assert loaded["ds"].dt.unit == "s"  # not promoted to [ms] by parquet
        assert str(loaded["ds"].dt.tz) == "US/Eastern"


def test_manifest_without_ds_unit_still_loads(tmp_path):
    """Snapshots written before ``ds_unit`` existed must still be readable.

    ``_restore_ds_unit`` is a no-op when the manifest entry carries no
    ``ds_unit`` key; this pins that backward compatibility so a store on disk
    from an older release keeps loading after the tz fix.
    """
    import json

    store = SnapshotStore(tmp_path)
    store.snapshot(_panel([1, 2, 3, 4]), as_of="2024-02-15")

    manifest_path = tmp_path / "manifest.json"
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in entries:
        entry.pop("ds_unit", None)
    manifest_path.write_text(json.dumps(entries), encoding="utf-8")

    loaded = store.load()
    assert set(loaded["y"]) == {1.0, 2.0, 3.0, 4.0}
    assert pd.api.types.is_datetime64_any_dtype(loaded["ds"])


class TestManifestAtomicity:
    """``manifest.json`` must never be destroyed by a failed rewrite.

    ``_write_manifest`` used to open the real manifest with mode ``'w'``
    (truncate) and re-dump the whole entry list on every ``snapshot()``, so a
    crash / ENOSPC mid-``json.dump`` left a half-written manifest on disk and
    every read API (``load``/``history``/``manifest``/``as_of_dates``) raised
    ``corrupt snapshot manifest`` — orphaning every snapshot already in the
    store even though all the parquet files were intact. The correct behaviour
    for an append-only audit trail is all-or-nothing: a failed write leaves the
    previous complete manifest in place.
    """

    def test_failed_manifest_write_leaves_the_previous_manifest_intact(
        self, tmp_path, monkeypatch
    ):
        store = SnapshotStore(tmp_path)
        store.snapshot(_panel([1, 2, 3, 4]), as_of="2024-01-15")
        store.snapshot(_panel([5, 6, 7, 8]), as_of="2024-02-15")
        before = store.manifest()
        raw_before = (tmp_path / "manifest.json").read_bytes()

        def boom(obj, fp, **kwargs):
            fp.write('[{"as_of": "2024-')  # partial write, then die
            raise OSError("No space left on device")

        monkeypatch.setattr(store_module.json, "dump", boom)
        with pytest.raises(OSError):
            store.snapshot(_panel([9, 9, 9, 9]), as_of="2024-03-15")
        monkeypatch.undo()

        # the manifest on disk is byte-identical to the pre-crash one
        assert (tmp_path / "manifest.json").read_bytes() == raw_before
        # ... and every read API still works
        pd.testing.assert_frame_equal(store.manifest(), before)
        assert store.as_of_dates() == [
            pd.Timestamp("2024-01-15"),
            pd.Timestamp("2024-02-15"),
        ]
        assert set(store.load(as_of="2024-02-15")["y"]) == {5.0, 6.0, 7.0, 8.0}
        assert len(store.history()) == 8

    def test_no_temp_file_is_left_behind(self, tmp_path, monkeypatch):
        store = SnapshotStore(tmp_path)
        store.snapshot(_panel([1, 2, 3, 4]), as_of="2024-01-15")
        # a successful write leaves only manifest.json
        assert sorted(p.name for p in tmp_path.glob("manifest*")) == ["manifest.json"]

        def boom(obj, fp, **kwargs):
            raise OSError("No space left on device")

        monkeypatch.setattr(store_module.json, "dump", boom)
        with pytest.raises(OSError):
            store.snapshot(_panel([2, 2, 2, 2]), as_of="2024-02-15")
        monkeypatch.undo()
        assert sorted(p.name for p in tmp_path.glob("manifest*")) == ["manifest.json"]


class TestHistoryLatestOnly:
    """``history()`` stacks every revision of a re-snapshotted ``as_of``.

    That is deliberate (the store is an audit trail) but there was no way to
    ask for the latest-per-``as_of`` view that ``load()`` gives, so feeding
    ``history()`` into ``accuracy_over_time`` double-counted a revised as-of.
    ``latest_only=True`` opts into ``load()``'s "last manifest match wins"
    rule; the default keeps the historical stacking behaviour.
    """

    def test_default_still_stacks_every_revision(self, tmp_path):
        store = SnapshotStore(tmp_path)
        store.snapshot(_panel([1, 1, 1, 1]), as_of="2024-02-15")
        store.snapshot(_panel([9, 9, 9, 9]), as_of="2024-02-15")
        hist = store.history()
        assert len(hist) == 8
        assert set(hist["y"]) == {1.0, 9.0}

    def test_latest_only_keeps_one_snapshot_per_as_of(self, tmp_path):
        store = SnapshotStore(tmp_path)
        store.snapshot(_panel([1, 1, 1, 1]), as_of="2024-02-15")
        store.snapshot(_panel([9, 9, 9, 9]), as_of="2024-02-15")
        store.snapshot(_panel([2, 2, 2, 2]), as_of="2024-03-15")

        hist = store.history(latest_only=True)
        assert len(hist) == 8  # two as_of dates x 4 rows, not three snapshots
        assert set(pd.to_datetime(hist["as_of"].unique())) == {
            pd.Timestamp("2024-02-15"),
            pd.Timestamp("2024-03-15"),
        }
        # the revision wins for 2024-02-15, matching load()
        feb = hist[hist["as_of"] == pd.Timestamp("2024-02-15")]
        assert set(feb["y"]) == {9.0}
        assert set(feb["y"]) == set(store.load(as_of="2024-02-15")["y"])

    def test_latest_only_composes_with_the_series_filter(self, tmp_path):
        store = SnapshotStore(tmp_path)
        store.snapshot(_panel([1, 1, 1, 1]), as_of="2024-02-15")
        store.snapshot(_panel([9, 9, 9, 9]), as_of="2024-02-15")
        hist = store.history(series="a", latest_only=True)
        assert set(hist["unique_id"]) == {"a"}
        assert set(hist["y"]) == {9.0}

    def test_latest_only_on_an_empty_store_returns_an_empty_frame(self, tmp_path):
        store = SnapshotStore(tmp_path)
        assert store.history(latest_only=True).empty
