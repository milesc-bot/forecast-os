"""Pilot-harness tests for the terminal app (textual + pytest-asyncio).

The import-guard test runs without textual (the module imports cleanly and
``main`` raises the install hint when the flag says the extra is missing);
the pilot tests drive a headless demo app and are skipped when the optional
``[terminal]`` extra is not installed.
"""


import pandas as pd
import pytest

from forecast_os.terminal import app as app_module
from forecast_os.terminal import engine_bridge
from forecast_os.terminal.workspace import Workspace

needs_textual = pytest.mark.skipif(
    not app_module._HAS_TEXTUAL, reason="optional terminal extra not installed"
)

DEMO_SERIES_COUNT = 6


def make_demo_app(tmp_path):
    """A demo-config app: empty workspace (demo panel fallback), tmp home."""
    return app_module.ForecastOSApp(Workspace(home=tmp_path), demo=True)


def short_panel():
    """One 3-row series: too short for the CV span board/governance need."""
    return pd.DataFrame(
        {
            "unique_id": "tiny/one",
            "ds": pd.date_range("2026-01-01", periods=3, freq="MS"),
            "y": [10.0, 11.0, 12.0],
        }
    )


class TestImportGuard:
    def test_module_imports_and_main_raises_hint_without_textual(self, monkeypatch):
        monkeypatch.setattr(app_module, "_HAS_TEXTUAL", False)
        with pytest.raises(ImportError, match=r"forecast-os\[terminal\]"):
            app_module.main(["--demo"])

    def test_parser_accepts_documented_flags(self):
        args = app_module._build_parser().parse_args(
            ["--demo", "--data", "a.csv", "--data", "b.csv", "--mapping", "generic_events"]
        )
        assert args.demo and args.data == ["a.csv", "b.csv"]
        assert args.mapping == "generic_events"


@needs_textual
class TestPilot:
    @pytest.mark.asyncio
    async def test_boots_to_dashboard_in_demo_config(self, tmp_path):
        app = make_demo_app(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert type(app.screen).__name__ == "DashboardScreen"
            assert "home" in app.sub_title  # the status line is live
            await app.workers.wait_for_complete()

    @pytest.mark.asyncio
    async def test_keys_switch_screens(self, tmp_path):
        app = make_demo_app(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            for key, expected in [
                ("f", "ForecastScreen"),
                ("l", "LeaderboardScreen"),
                ("g", "GovernanceScreen"),
                ("s", "SourcesScreen"),
                ("d", "DashboardScreen"),
            ]:
                await pilot.press(key)
                await pilot.pause()
                assert type(app.screen).__name__ == expected
            await app.workers.wait_for_complete()

    @pytest.mark.asyncio
    async def test_refresh_fills_dashboard_table(self, tmp_path):
        from textual.widgets import DataTable

        app = make_demo_app(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("r")
            await app.workers.wait_for_complete()
            await pilot.pause()
            table = app.screen.query_one("#dashboard-table", DataTable)
            assert table.row_count == DEMO_SERIES_COUNT
            assert app.panel is not None
            assert app.last_refresh is not None
            assert "refreshed" in app.sub_title

    @pytest.mark.asyncio
    async def test_quit_key_exits(self, tmp_path):
        app = make_demo_app(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await app.workers.wait_for_complete()
            await pilot.press("q")
        assert not app.is_running


def _write_panel_csv(tmp_path, panel, name="src.csv"):
    path = tmp_path / name
    panel.to_csv(path, index=False)
    return path


@needs_textual
class TestFailureHandling:
    """Verifier regressions: no retry hot loop, no worker-crash, visible errors."""

    @pytest.mark.asyncio
    async def test_short_series_leaderboard_shows_unavailable_no_hot_loop(
        self, tmp_path, monkeypatch
    ):
        calls = {"n": 0}
        real = engine_bridge.leaderboard_frame

        def counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(engine_bridge, "leaderboard_frame", counting)
        csv = _write_panel_csv(tmp_path, short_panel())
        source = {"path": str(csv), "mapping": None, "overrides": {}}
        ws = Workspace(home=tmp_path, sources=[source])
        app = app_module.ForecastOSApp(ws, demo=False)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("r")
            await app.workers.wait_for_complete()
            await pilot.press("l")
            await app.workers.wait_for_complete()
            await pilot.pause()
            # the compute is attempted a bounded number of times, not thousands
            assert calls["n"] <= 2, f"leaderboard recomputed {calls['n']} times (hot loop)"
            assert isinstance(app.board, engine_bridge.ComputeFailure)
            # app survives: dashboard still switchable
            await pilot.press("d")
            await pilot.pause()
            assert type(app.screen).__name__ == "DashboardScreen"

    @pytest.mark.asyncio
    async def test_short_series_governance_no_crash_no_hot_loop(self, tmp_path, monkeypatch):
        calls = {"n": 0}
        real = engine_bridge.governance_frame

        def counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(engine_bridge, "governance_frame", counting)
        csv = _write_panel_csv(tmp_path, short_panel())
        source = {"path": str(csv), "mapping": None, "overrides": {}}
        ws = Workspace(home=tmp_path, sources=[source])
        app = app_module.ForecastOSApp(ws, demo=False)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("r")
            await app.workers.wait_for_complete()
            await pilot.press("g")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert calls["n"] <= 2, f"governance recomputed {calls['n']} times (hot loop)"
            assert isinstance(app.governance, engine_bridge.ComputeFailure)
            assert app.is_running  # app not killed by WorkerFailed

    @pytest.mark.asyncio
    async def test_broken_source_error_visible_on_dashboard(self, tmp_path):
        from textual.widgets import Static

        ws = Workspace(
            home=tmp_path,
            sources=[{"path": str(tmp_path / "nope.csv"), "mapping": None, "overrides": {}}],
        )
        app = app_module.ForecastOSApp(ws, demo=False)
        async with app.run_test(size=(100, 30)) as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert any("refresh failed" in m for m in app.fired_alerts)
            alerts = app.screen.query_one("#dashboard-alerts", Static)
            assert "refresh failed" in str(alerts.render())

    @pytest.mark.asyncio
    async def test_rapid_refresh_no_duplicate_rows(self, tmp_path):
        from textual.widgets import DataTable

        app = make_demo_app(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(5):
                await pilot.press("r")
            await app.workers.wait_for_complete()
            await pilot.pause()
            table = app.screen.query_one("#dashboard-table", DataTable)
            assert table.row_count == DEMO_SERIES_COUNT  # exclusive worker → no dupes


@needs_textual
def test_main_reports_corrupt_workspace_as_clean_exit(tmp_path):
    (tmp_path / "workspace.json").write_text('{"settings": "oops"}')
    with pytest.raises(SystemExit) as exc:
        app_module.main(["--home", str(tmp_path)])
    assert "settings" in str(exc.value)
