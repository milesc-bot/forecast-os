"""Pilot-harness tests for the terminal app (textual + pytest-asyncio).

The import-guard test runs without textual (the module imports cleanly and
``main`` raises the install hint when the flag says the extra is missing);
the pilot tests drive a headless demo app and are skipped when the optional
``[terminal]`` extra is not installed.
"""

import asyncio
import threading

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
class TestRefreshInvalidatesInFlightWorkers:
    """Regression: an explicit refresh must not be overwritten by a stale worker.

    What went wrong: ``action_refresh`` set ``board``/``governance`` back to
    ``None``, but the already-running board/governance worker lives in a
    different worker group ("board"/"governance") than the refresh worker
    ("refresh"), so ``exclusive=True`` never cancelled it — and a thread
    worker cannot be cancelled mid-run anyway. It finished and wrote its
    PRE-refresh result over the ``None``. Because ``request_board`` only
    fires ``if self.board is None``, nothing ever recomputed it, so pressing
    ``r`` (the leaderboard hint literally reads "r recomputes") left a stale
    board on screen with no indication it was stale.

    The right behaviour: a result produced by a worker that started before
    the refresh is discarded, and the board/governance is recomputed from the
    post-refresh panel.
    """

    @pytest.mark.asyncio
    async def test_refresh_discards_in_flight_board(self, tmp_path, monkeypatch):
        calls = {"n": 0}
        gate = threading.Event()

        def staged(panel, settings, *a, **k):
            calls["n"] += 1
            n = calls["n"]
            if n == 1:  # the pre-refresh run: still in flight when "r" is pressed
                gate.wait(timeout=10)
            return pd.DataFrame({"model": [f"v{n}"], "mase": [float(n)]})

        monkeypatch.setattr(engine_bridge, "leaderboard_frame", staged)
        app = make_demo_app(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await app.workers.wait_for_complete()
            await pilot.press("l")  # requests the board; worker 1 blocks on the gate
            await pilot.pause()
            assert app._board_working
            await pilot.press("r")  # user asks for a recompute
            gate.set()  # ...and only now does the pre-refresh worker finish
            for _ in range(80):
                await pilot.pause()
                await asyncio.sleep(0.02)
                if calls["n"] >= 2 and not app._board_working:
                    break
            assert app.board is not None, "board never recomputed after refresh"
            assert list(app.board["model"]) == ["v2"], (
                f"refresh kept the pre-refresh board {list(app.board['model'])}"
            )

    @pytest.mark.asyncio
    async def test_refresh_discards_in_flight_governance(self, tmp_path, monkeypatch):
        calls = {"n": 0}
        gate = threading.Event()

        def staged(panel, settings, *a, **k):
            calls["n"] += 1
            n = calls["n"]
            if n == 1:
                gate.wait(timeout=10)
            return pd.DataFrame(
                {
                    "unique_id": [f"v{n}"],
                    "cutoff": [pd.Timestamp("2026-01-01")],
                    "mase": [float(n)],
                    "pct_bias": [0.0],
                    "coverage": [1.0],
                }
            )

        monkeypatch.setattr(engine_bridge, "governance_frame", staged)
        app = make_demo_app(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await app.workers.wait_for_complete()
            await pilot.press("g")
            await pilot.pause()
            assert app._governance_working
            await pilot.press("r")
            gate.set()
            for _ in range(80):
                await pilot.pause()
                await asyncio.sleep(0.02)
                if calls["n"] >= 2 and not app._governance_working:
                    break
            assert app.governance is not None, "governance never recomputed after refresh"
            assert list(app.governance["unique_id"]) == ["v2"], (
                f"refresh kept the pre-refresh governance "
                f"{list(app.governance['unique_id'])}"
            )

    @pytest.mark.asyncio
    async def test_stale_refresh_does_not_reinstate_an_older_panel(
        self, tmp_path, monkeypatch
    ):
        """The refresh worker needs the same generation guard as board/governance.

        ``_refresh_worker`` is ``exclusive=True, group="refresh"``, but a
        thread worker cannot be cancelled mid-run: a slow refresh started
        before the user pressed ``r`` keeps going and used to overwrite
        ``panel``/``rows``/``fired_alerts``/``last_refresh`` — and assigned
        ``self.governance`` directly, walking a pre-refresh governance frame
        straight past the ``_apply_governance`` guard. The console then showed
        data older than its own "refreshed HH:MM:SS" stamp.
        """

        def panel_for(tag):
            return pd.DataFrame(
                {
                    "unique_id": [tag] * 40,
                    "ds": pd.date_range("2020-01-01", periods=40, freq="MS"),
                    "y": [float(i + 1) for i in range(40)],
                }
            )

        calls = {"n": 0}
        gate = threading.Event()

        def staged_refresh(self):
            calls["n"] += 1
            if calls["n"] == 1:
                gate.wait(timeout=10)  # the older refresh lands last
                return panel_for("old")
            return panel_for("new")

        monkeypatch.setattr(app_module.PanelProvider, "refresh", staged_refresh)
        app = make_demo_app(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(50):  # on_mount fired refresh #1; it blocks on the gate
                await pilot.pause()
                await asyncio.sleep(0.01)
                if calls["n"] >= 1:
                    break
            await pilot.press("r")  # refresh #2 overtakes it
            for _ in range(100):
                await pilot.pause()
                await asyncio.sleep(0.02)
                if calls["n"] >= 2 and app.panel is not None:
                    break
            assert app.panel["unique_id"].iloc[0] == "new"
            gate.set()  # ...and only now does the superseded refresh finish
            for _ in range(100):
                await pilot.pause()
                await asyncio.sleep(0.02)
                if app.panel["unique_id"].iloc[0] == "old":
                    break
            assert app.panel["unique_id"].iloc[0] == "new", (
                "a superseded refresh reinstated its older panel"
            )


@needs_textual
def test_main_reports_corrupt_workspace_as_clean_exit(tmp_path):
    (tmp_path / "workspace.json").write_text('{"settings": "oops"}')
    with pytest.raises(SystemExit) as exc:
        app_module.main(["--home", str(tmp_path)])
    assert "settings" in str(exc.value)
