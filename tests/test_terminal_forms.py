"""Pilot-harness tests for source/alert editing forms and dashboard drill-down.

The import-guard test runs without textual (the modals module imports cleanly
and its modal factories raise the install hint when the flag says the extra is
missing); the pilot tests drive a headless app and are skipped when the
optional ``[terminal]`` extra is not installed. They mirror the patterns in
``tests/test_terminal_app.py`` and stay robust to worker timing by awaiting
the worker pool after every mutation.
"""


import pandas as pd
import pytest

from forecast_os.terminal import app as app_module
from forecast_os.terminal.screens import modals as modals_module
from forecast_os.terminal.workspace import Workspace

needs_textual = pytest.mark.skipif(
    not app_module._HAS_TEXTUAL, reason="optional terminal extra not installed"
)

DEMO_SERIES_COUNT = 6


def _panel_csv(tmp_path, name="src.csv"):
    """Write a small valid contract-column CSV; return its path (as str)."""
    path = tmp_path / name
    pd.DataFrame(
        {
            "unique_id": ["x"] * 4,
            "ds": pd.date_range("2026-01-01", periods=4, freq="MS"),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_csv(path, index=False)
    return str(path)


def make_demo_app(tmp_path):
    """A demo-config app: empty workspace (demo panel fallback), tmp home."""
    return app_module.ForecastOSApp(Workspace(home=tmp_path), demo=True)


class TestModalsImportGuard:
    def test_module_imports_and_exposes_flag(self):
        # The module imports without touching textual at import time.
        assert hasattr(modals_module, "_HAS_TEXTUAL")

    def test_factories_raise_hint_without_textual(self, monkeypatch):
        monkeypatch.setattr(modals_module, "_HAS_TEXTUAL", False)
        with pytest.raises(ImportError, match=r"forecast-os\[terminal\]"):
            modals_module.new_source_modal()
        with pytest.raises(ImportError, match=r"forecast-os\[terminal\]"):
            modals_module.new_alert_modal()


async def _goto_sources(pilot, app):
    await app.workers.wait_for_complete()
    await pilot.press("s")
    await pilot.pause()
    assert type(app.screen).__name__ == "SourcesScreen"


@needs_textual
class TestSourceForms:
    @pytest.mark.asyncio
    async def test_add_source_lands_and_persists(self, tmp_path):
        from textual.widgets import Input

        csv = _panel_csv(tmp_path)
        ws = Workspace(home=tmp_path)
        app = app_module.ForecastOSApp(ws, demo=False)
        async with app.run_test(size=(120, 40)) as pilot:
            await _goto_sources(pilot, app)
            await pilot.press("a")
            await pilot.pause()
            assert type(app.screen).__name__ == "SourceModal"
            app.screen.query_one("#source-path", Input).value = csv
            app.screen.action_submit()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
        # landed in memory with the documented shape
        assert len(ws.sources) == 1
        src = ws.sources[0]
        assert src == {"path": csv, "mapping": None, "overrides": {}}
        # persisted: a fresh load from the same home sees it
        fresh = Workspace.load(home=tmp_path)
        assert any(s["path"] == csv for s in fresh.sources)

    @pytest.mark.asyncio
    async def test_add_source_with_mapping_and_freq_override(self, tmp_path):
        from textual.widgets import Input, Select

        ws = Workspace(home=tmp_path)
        app = app_module.ForecastOSApp(ws, demo=False)
        async with app.run_test(size=(120, 40)) as pilot:
            await _goto_sources(pilot, app)
            await pilot.press("a")
            await pilot.pause()
            app.screen.query_one("#source-path", Input).value = "raw.csv"
            app.screen.query_one("#source-mapping", Select).value = "generic_events"
            app.screen.query_one("#source-freq", Input).value = "W"
            app.screen.action_submit()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
        assert ws.sources[0]["mapping"] == "generic_events"
        assert ws.sources[0]["overrides"] == {"freq": "W"}

    @pytest.mark.asyncio
    async def test_edit_source_updates_and_persists(self, tmp_path):
        from textual.widgets import Input

        csv1 = _panel_csv(tmp_path, "one.csv")
        csv2 = _panel_csv(tmp_path, "two.csv")
        ws = Workspace(
            home=tmp_path,
            sources=[{"path": csv1, "mapping": None, "overrides": {}}],
        )
        app = app_module.ForecastOSApp(ws, demo=False)
        async with app.run_test(size=(120, 40)) as pilot:
            await _goto_sources(pilot, app)
            await pilot.press("e")
            await pilot.pause()
            assert type(app.screen).__name__ == "SourceModal"
            path_input = app.screen.query_one("#source-path", Input)
            assert path_input.value == csv1  # pre-filled from the source
            path_input.value = csv2
            app.screen.action_submit()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
        assert len(ws.sources) == 1
        assert ws.sources[0]["path"] == csv2
        fresh = Workspace.load(home=tmp_path)
        assert fresh.sources[0]["path"] == csv2

    @pytest.mark.asyncio
    async def test_delete_source_persists(self, tmp_path):
        csv = _panel_csv(tmp_path)
        ws = Workspace(
            home=tmp_path,
            sources=[{"path": csv, "mapping": None, "overrides": {}}],
        )
        app = app_module.ForecastOSApp(ws, demo=False)
        async with app.run_test(size=(120, 40)) as pilot:
            await _goto_sources(pilot, app)
            from textual.widgets import DataTable

            table = app.screen.query_one("#sources-table", DataTable)
            table.move_cursor(row=0)
            await pilot.press("x")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
        assert ws.sources == []
        fresh = Workspace.load(home=tmp_path)
        assert fresh.sources == []

    @pytest.mark.asyncio
    async def test_demo_add_source_not_written(self, tmp_path):
        from textual.widgets import Input

        csv = _panel_csv(tmp_path)
        app = make_demo_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _goto_sources(pilot, app)
            await pilot.press("a")
            await pilot.pause()
            app.screen.query_one("#source-path", Input).value = csv
            app.screen.action_submit()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
        # appended in memory, but --demo writes nothing to disk
        assert any(s["path"] == csv for s in app.workspace.sources)
        assert not (tmp_path / "workspace.json").exists()


@needs_textual
class TestAlertForms:
    @pytest.mark.asyncio
    async def test_add_and_delete_alert_round_trips(self, tmp_path):
        from textual.widgets import DataTable, Input

        ws = Workspace(home=tmp_path)
        app = app_module.ForecastOSApp(ws, demo=False)
        async with app.run_test(size=(120, 40)) as pilot:
            await _goto_sources(pilot, app)
            await pilot.press("A")
            await pilot.pause()
            assert type(app.screen).__name__ == "AlertModal"
            app.screen.query_one("#alert-threshold", Input).value = "50"
            app.screen.action_submit()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert ws.alerts == [
                {"kind": "forecast_below", "series": "*", "threshold": 50.0}
            ]
            # persisted
            assert Workspace.load(home=tmp_path).alerts[0]["threshold"] == 50.0
            # delete the selected alert
            table = app.screen.query_one("#alerts-table", DataTable)
            table.move_cursor(row=0)
            await pilot.press("X")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert ws.alerts == []
        assert Workspace.load(home=tmp_path).alerts == []

    @pytest.mark.asyncio
    async def test_alert_bad_threshold_does_not_dismiss(self, tmp_path):
        from textual.widgets import Input

        ws = Workspace(home=tmp_path)
        app = app_module.ForecastOSApp(ws, demo=False)
        async with app.run_test(size=(120, 40)) as pilot:
            await _goto_sources(pilot, app)
            await pilot.press("A")
            await pilot.pause()
            app.screen.query_one("#alert-threshold", Input).value = "not-a-number"
            app.screen.action_submit()
            await pilot.pause()
            # invalid threshold: the modal stays open, nothing added
            assert type(app.screen).__name__ == "AlertModal"
            assert ws.alerts == []


@needs_textual
class TestDrillDown:
    @pytest.mark.asyncio
    async def test_dashboard_enter_selects_series_in_forecast(self, tmp_path):
        from textual.widgets import DataTable, ListView

        app = make_demo_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("r")
            await app.workers.wait_for_complete()
            await pilot.pause()
            table = app.screen.query_one("#dashboard-table", DataTable)
            assert table.row_count == DEMO_SERIES_COUNT
            app.screen.set_focus(table)
            table.move_cursor(row=2)
            target = table.get_row_at(2)[0]
            await pilot.press("enter")
            for _ in range(6):
                await pilot.pause()
            assert type(app.screen).__name__ == "ForecastScreen"
            listview = app.screen.query_one("#series-list", ListView)
            item = listview.highlighted_child
            assert item is not None
            assert item.name == target
