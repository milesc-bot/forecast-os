"""Tests for the textual-free terminal layers: workspace, data provider, engine bridge."""

import json

import numpy as np
import pandas as pd
import pytest

from forecast_os.core.exceptions import ForecastOSError
from forecast_os.core.types import ID_COL, TARGET_COL, TIME_COL, validate_panel
from forecast_os.terminal import engine_bridge
from forecast_os.terminal.data import PanelProvider, demo_panel
from forecast_os.terminal.workspace import Workspace, default_settings

DEMO_UIDS = ["east/dan", "east/erin", "east/fay", "west/alice", "west/bob", "west/cara"]


@pytest.fixture
def demo():
    return demo_panel()


@pytest.fixture
def settings():
    return default_settings()


def contract_csv(path, uid, n=24, start="2024-01-01"):
    """Write a contract-shaped (unique_id, ds, y) CSV; return its path."""
    ds = pd.date_range(start, periods=n, freq="MS")
    frame = pd.DataFrame(
        {ID_COL: uid, TIME_COL: ds.strftime("%Y-%m-%d"), TARGET_COL: 50.0 + np.arange(n)}
    )
    frame.to_csv(path, index=False)
    return path


class TestWorkspace:
    def test_load_without_file_gives_defaults(self, tmp_path):
        ws = Workspace.load(home=tmp_path)
        assert ws.home == tmp_path
        assert ws.sources == []
        assert ws.watch == []
        assert ws.alerts == []
        assert ws.settings == {
            "model": "auto_ets",
            "h": 6,
            "level": 80,
            "refresh_seconds": 0,
            "season_length": None,
        }

    def test_round_trip(self, tmp_path):
        ws = Workspace.load(home=tmp_path)
        ws.sources.append({"path": "deals.csv", "mapping": "hubspot_deals", "overrides": {}})
        ws.watch = ["west/alice"]
        ws.settings["h"] = 12
        ws.settings["season_length"] = 3
        ws.alerts.append({"kind": "forecast_below", "series": "*", "threshold": 40.0})
        target = ws.save()
        assert target == tmp_path / "workspace.json"
        assert not target.with_name(target.name + ".tmp").exists()  # atomic rename

        back = Workspace.load(home=tmp_path)
        assert back.sources == ws.sources
        assert back.watch == ws.watch
        assert back.settings == ws.settings
        assert back.alerts == ws.alerts

    def test_env_var_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FORECAST_OS_HOME", str(tmp_path / "envhome"))
        ws = Workspace.load()
        assert ws.home == tmp_path / "envhome"
        # an explicit home wins over the env var
        assert Workspace.load(home=tmp_path).home == tmp_path

    def test_tolerant_merge_of_partial_file(self, tmp_path):
        (tmp_path / "workspace.json").write_text(json.dumps({"settings": {"h": 9}}))
        ws = Workspace.load(home=tmp_path)
        assert ws.settings["h"] == 9
        assert ws.settings["model"] == "auto_ets"  # missing keys fall back
        assert ws.sources == [] and ws.watch == [] and ws.alerts == []

    def test_partial_source_dicts_get_defaults(self, tmp_path):
        (tmp_path / "workspace.json").write_text(
            json.dumps({"sources": [{"path": "a.csv"}]})
        )
        ws = Workspace.load(home=tmp_path)
        assert ws.sources == [{"path": "a.csv", "mapping": None, "overrides": {}}]

    def test_corrupt_file_raises(self, tmp_path):
        (tmp_path / "workspace.json").write_text("{not json")
        with pytest.raises(ForecastOSError, match="workspace"):
            Workspace.load(home=tmp_path)

    def test_save_without_home_raises(self):
        with pytest.raises(ForecastOSError, match="home"):
            Workspace().save()


class TestDemoPanel:
    def test_validates_and_has_expected_shape(self, demo):
        validate_panel(demo)
        assert sorted(demo[ID_COL].unique()) == DEMO_UIDS
        assert demo.groupby(ID_COL).size().eq(30).all()
        assert (demo[TARGET_COL] >= 0).all()

    def test_deterministic(self, demo):
        pd.testing.assert_frame_equal(demo, demo_panel())


class TestPanelProvider:
    def test_empty_workspace_falls_back_to_demo(self, tmp_path, demo):
        provider = PanelProvider(Workspace.load(home=tmp_path))
        pd.testing.assert_frame_equal(provider.load(), demo)

    def test_multi_source_concat(self, tmp_path):
        ws = Workspace.load(home=tmp_path)
        for uid in ("acme/a", "acme/b"):
            path = contract_csv(tmp_path / f"{uid.replace('/', '-')}.csv", uid)
            ws.sources.append({"path": str(path), "mapping": None, "overrides": {}})
        panel = PanelProvider(ws).load()
        validate_panel(panel)
        assert sorted(panel[ID_COL].unique()) == ["acme/a", "acme/b"]
        assert len(panel) == 48

    def test_mapping_source(self, tmp_path):
        events = pd.DataFrame({"date": ["2026-01-01", "2026-01-01", "2026-01-02"]})
        path = tmp_path / "events.csv"
        events.to_csv(path, index=False)
        ws = Workspace.load(home=tmp_path)
        ws.sources.append({"path": str(path), "mapping": "generic_events", "overrides": {}})
        panel = PanelProvider(ws).load()
        assert panel[TARGET_COL].tolist() == [2.0, 1.0]  # daily row counts

    def test_failing_source_names_path_and_reason(self, tmp_path):
        ws = Workspace.load(home=tmp_path)
        ws.sources.append({"path": str(tmp_path / "nope.csv"), "mapping": None, "overrides": {}})
        with pytest.raises(ForecastOSError, match="nope.csv"):
            PanelProvider(ws).load()

    def test_source_without_contract_columns_names_path(self, tmp_path):
        pd.DataFrame({"a": [1]}).to_csv(tmp_path / "bad.csv", index=False)
        ws = Workspace.load(home=tmp_path)
        ws.sources.append({"path": str(tmp_path / "bad.csv"), "mapping": None, "overrides": {}})
        with pytest.raises(ForecastOSError, match=r"bad.csv.*unique_id"):
            PanelProvider(ws).load()

    def test_refresh_rereads(self, tmp_path):
        path = contract_csv(tmp_path / "one.csv", "acme/a", n=12)
        ws = Workspace.load(home=tmp_path)
        ws.sources.append({"path": str(path), "mapping": None, "overrides": {}})
        provider = PanelProvider(ws)
        assert len(provider.load()) == 12
        contract_csv(path, "acme/a", n=13)
        assert len(provider.refresh()) == 13


class TestSparkline:
    def test_length_and_alphabet(self):
        spark = engine_bridge.sparkline(range(30))
        assert len(spark) == 12
        assert set(spark) <= set("▁▂▃▄▅▆▇█")
        assert spark[0] == "▁" and spark[-1] == "█"

    def test_short_constant_and_nan_input(self):
        assert len(engine_bridge.sparkline([1.0, 2.0])) == 2
        assert engine_bridge.sparkline([5.0] * 4) == "▄▄▄▄"
        assert "·" in engine_bridge.sparkline([1.0, float("nan"), 2.0])
        assert engine_bridge.sparkline([]) == ""


class TestModelParams:
    def test_season_length_passed_only_when_accepted(self, settings):
        settings["season_length"] = 3
        assert engine_bridge.model_params("theta", settings) == {"season_length": 3}
        assert engine_bridge.model_params("auto_ets", settings) == {"season_length": 3}
        assert engine_bridge.model_params("naive", settings) == {}

    def test_none_season_length_passes_nothing(self, settings):
        assert engine_bridge.model_params("theta", settings) == {}


class TestDashboardRows:
    def test_shape_and_columns(self, demo, settings):
        rows = engine_bridge.dashboard_rows(demo, settings)
        assert list(rows.columns) == [ID_COL, "last", "next", "delta_pct", "spark"]
        assert rows[ID_COL].tolist() == DEMO_UIDS
        assert rows["spark"].str.len().eq(12).all()
        assert np.isfinite(rows[["last", "next", "delta_pct"]].to_numpy()).all()
        last = demo[demo[ID_COL] == "west/alice"][TARGET_COL].iloc[-1]
        assert rows.set_index(ID_COL).loc["west/alice", "last"] == pytest.approx(last)

    def test_poisoned_series_falls_back_to_naive(self, demo, settings):
        bad = pd.DataFrame(
            {
                ID_COL: "poison/one",
                TIME_COL: pd.date_range("2024-01-01", periods=5, freq="MS"),
                TARGET_COL: [10.0, float("nan"), 12.0, 13.0, 14.0],
            }
        )
        rows = engine_bridge.dashboard_rows(pd.concat([demo, bad]), settings)
        assert len(rows) == 7  # the poisoned series did not sink the board
        row = rows.set_index(ID_COL).loc["poison/one"]
        assert row["next"] == pytest.approx(row["last"])  # naive fallback


class TestForecastFrame:
    def test_history_and_forecast_shapes(self, demo, settings):
        history, forecast = engine_bridge.forecast_frame(demo, "west/alice", settings)
        assert list(history.columns) == [ID_COL, TIME_COL, TARGET_COL]
        assert len(history) == 30
        assert len(forecast) == settings["h"]
        assert {"yhat", "lo", "hi"} <= set(forecast.columns)
        assert (forecast["lo"] <= forecast["hi"]).all()
        assert forecast[TIME_COL].iloc[0] > history[TIME_COL].iloc[-1]

    def test_unknown_series_raises(self, demo, settings):
        with pytest.raises(ForecastOSError, match="unknown series"):
            engine_bridge.forecast_frame(demo, "nope/nobody", settings)


class TestLeaderboardFrame:
    def test_board_shape(self, demo, settings):
        board = engine_bridge.leaderboard_frame(demo, settings)
        assert "model" in board.columns
        assert {"mase", "coverage-80"} <= set(board.columns)
        assert len(board) == 4  # every default candidate survives on the demo
        assert (board["mase"] > 0).all()
        # compare() sorts by the first metric: mase ascending
        assert board["mase"].is_monotonic_increasing

    def test_explicit_model_list(self, demo, settings):
        board = engine_bridge.leaderboard_frame(demo, settings, models=["naive", "theta"])
        assert set(board["model"]) == {"naive", "theta"}


class TestGovernanceFrame:
    def test_tidy_table(self, demo, settings):
        gov = engine_bridge.governance_frame(demo, settings)
        assert list(gov.columns) == [ID_COL, "cutoff", "mase", "pct_bias", "coverage"]
        assert len(gov) == 6 * 2  # six series, two cutoffs
        assert gov["coverage"].between(0.0, 1.0).all()
        assert np.isfinite(gov["pct_bias"]).all()


class TestAlertsFired:
    def test_forecast_below_star_and_single_series(self, demo, settings):
        forecasts = engine_bridge.dashboard_rows(demo, settings).rename(
            columns={"next": "yhat"}
        )
        everything = engine_bridge.alerts_fired(
            demo, forecasts, [{"kind": "forecast_below", "series": "*", "threshold": 1e9}]
        )
        assert len(everything) == 6
        assert all("forecast_below" in msg for msg in everything)
        one = engine_bridge.alerts_fired(
            demo,
            forecasts,
            [{"kind": "forecast_below", "series": "east/fay", "threshold": 1e9}],
        )
        assert len(one) == 1 and "east/fay" in one[0]
        assert (
            engine_bridge.alerts_fired(
                demo, forecasts, [{"kind": "forecast_below", "series": "*", "threshold": 0.0}]
            )
            == []
        )

    def test_coverage_below_uses_governance(self, demo, settings):
        forecasts = pd.DataFrame({ID_COL: DEMO_UIDS, "yhat": 100.0})
        governance = pd.DataFrame(
            {ID_COL: DEMO_UIDS, "cutoff": "2026-01-01", "coverage": [0.9, 0.9, 0.5, 0.9, 0.9, 0.9]}
        )
        fired = engine_bridge.alerts_fired(
            demo,
            forecasts,
            [{"kind": "coverage_below", "series": "*", "threshold": 0.65}],
            governance=governance,
        )
        assert len(fired) == 1 and "east/fay" in fired[0]

    def test_coverage_below_without_governance_raises(self, demo):
        forecasts = pd.DataFrame({ID_COL: DEMO_UIDS, "yhat": 100.0})
        with pytest.raises(ValueError, match="governance"):
            engine_bridge.alerts_fired(
                demo, forecasts, [{"kind": "coverage_below", "series": "*", "threshold": 0.5}]
            )

    def test_unknown_kind_raises_and_empty_rules_no_op(self, demo):
        forecasts = pd.DataFrame({ID_COL: DEMO_UIDS, "yhat": 100.0})
        assert engine_bridge.alerts_fired(demo, forecasts, []) == []
        with pytest.raises(ValueError, match="unknown alert kind"):
            engine_bridge.alerts_fired(
                demo, forecasts, [{"kind": "wat", "series": "*", "threshold": 1.0}]
            )


class TestWorkspaceTypeValidation:
    """Corrupt/wrong-typed workspace fields raise ForecastOSError, not tracebacks."""

    @pytest.mark.parametrize(
        "payload,field",
        [
            ({"settings": "oops"}, "settings"),
            ({"sources": "oops"}, "sources"),
            ({"sources": [1, 2]}, "sources"),
            ({"watch": {"a": 1}}, "watch"),
            ({"alerts": "oops"}, "alerts"),
            ({"alerts": [1]}, "alerts"),
        ],
    )
    def test_wrong_typed_field_raises_named_error(self, tmp_path, payload, field):
        (tmp_path / "workspace.json").write_text(json.dumps(payload))
        with pytest.raises(ForecastOSError, match=field):
            Workspace.load(home=tmp_path)

    def test_valid_partial_file_still_merges(self, tmp_path):
        (tmp_path / "workspace.json").write_text(
            json.dumps({"settings": {"h": 12}, "watch": ["west/alice"]})
        )
        ws = Workspace.load(home=tmp_path)
        assert ws.settings["h"] == 12
        assert ws.settings["model"] == "auto_ets"  # default merged in
        assert ws.watch == ["west/alice"]
        assert ws.sources == [] and ws.alerts == []
