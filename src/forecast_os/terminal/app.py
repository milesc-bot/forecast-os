"""The always-on forecast console: ``forecast-os-tui``.

A keyboard-first textual app over the terminal compute layers: ``d``
dashboard, ``f`` forecast fan chart, ``l`` leaderboard, ``g`` governance,
``s`` sources, ``r`` refresh, ``q`` quit. All compute (panel reload,
dashboard rows, alerts, leaderboard, governance) runs in thread workers and
posts results back, so the UI never blocks; the header status line tracks
workspace home, panel shape, last refresh, and fired alerts.

The ``textual`` dependency is optional: importing this module always
succeeds, and :func:`main` raises ``ImportError`` with an install hint when
the ``[terminal]`` extra is missing.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from ..core.exceptions import ForecastOSError
from . import engine_bridge
from .data import PanelProvider
from .engine_bridge import ComputeFailure
from .workspace import Workspace

_TERMINAL_HINT = (
    "the terminal UI needs the optional textual dependency; "
    'install it with: pip install "forecast-os[terminal]"'
)

try:
    from textual import work
    from textual.app import App
    from textual.reactive import reactive
    from textual.widgets import ListView

    from .screens.dashboard import DashboardScreen
    from .screens.forecast import ForecastScreen
    from .screens.governance import GovernanceScreen
    from .screens.leaderboard import LeaderboardScreen
    from .screens.sources import SourcesScreen

    _HAS_TEXTUAL = True
except ImportError:  # tests exercise this path by monkeypatching the flag
    _HAS_TEXTUAL = False

__all__ = ["ForecastOSApp", "main"]


if _HAS_TEXTUAL:

    class ForecastOSApp(App):
        """The forecast-os console: five screens over one workspace panel."""

        TITLE = "forecast-os"

        CSS = """
        #dashboard-table, #leaderboard-table, #governance-table {
            height: 1fr;
        }
        #dashboard-alerts, #leaderboard-hint, #governance-hint, #sources-hint {
            height: auto;
            max-height: 6;
            padding: 0 1;
        }
        #sources-title, #alerts-title, #mappings-title {
            height: 1;
            padding: 0 1;
            background: $boost;
        }
        #sources-table, #alerts-table {
            height: auto;
            max-height: 8;
        }
        #mappings-table {
            height: 1fr;
        }
        #forecast-body {
            height: 1fr;
        }
        #series-list {
            width: 24;
            border-right: solid $primary;
        }
        #fan-chart {
            width: 1fr;
        }
        """

        BINDINGS = [
            ("d", "show_screen('dashboard')", "Dashboard"),
            ("f", "show_screen('forecast')", "Forecast"),
            ("l", "show_screen('leaderboard')", "Leaderboard"),
            ("g", "show_screen('governance')", "Governance"),
            ("s", "show_screen('sources')", "Sources"),
            ("r", "refresh", "Refresh"),
            ("ctrl+s", "save_workspace", "Save"),
            ("q", "quit", "Quit"),
        ]

        SCREENS = {
            "dashboard": DashboardScreen,
            "forecast": ForecastScreen,
            "leaderboard": LeaderboardScreen,
            "governance": GovernanceScreen,
            "sources": SourcesScreen,
        }

        refreshing: reactive[bool] = reactive(False)
        alerts_count: reactive[int] = reactive(0)

        def __init__(self, workspace: Workspace, demo: bool = False):
            super().__init__()
            self.workspace = workspace
            self.demo = demo
            self.provider = PanelProvider(workspace)
            self.panel = None
            self.rows = None
            self.fired_alerts: list[str] = []
            self.board = None
            self.governance = None
            self.last_refresh: str | None = None
            self._board_working = False
            self._governance_working = False
            # bumped by every refresh; workers capture it at start so a result
            # computed from the pre-refresh panel can be recognised and dropped
            self._generation = 0
            self._drill_target: str | None = None
            self._save_note: str | None = None

        # -- lifecycle ---------------------------------------------------

        def on_mount(self) -> None:
            self.push_screen("dashboard")
            self.action_refresh()
            seconds = int(self.workspace.settings.get("refresh_seconds") or 0)
            if seconds > 0:
                self.set_interval(seconds, self.action_refresh)

        # -- actions -----------------------------------------------------

        def action_show_screen(self, name: str) -> None:
            if type(self.screen) is not self.SCREENS[name]:
                self.switch_screen(name)

        def action_refresh(self) -> None:
            """Reload sources and recompute everything off the UI thread."""
            self._generation += 1
            self.board = None
            self.governance = None
            self.refreshing = True
            self._refresh_worker()

        def action_save_workspace(self) -> None:
            """Persist the workspace (bound to ctrl+s); a no-op on the demo panel."""
            self.save_workspace()
            self._update_status()

        def save_workspace(self) -> None:
            """Write the workspace unless running on the demo panel.

            Writes only when a home directory is set and the app is not in
            ``--demo`` mode; in demo mode nothing is written and a status note
            records that the save was skipped. Screens call this after every
            source/alert mutation.
            """
            if self.demo or self.workspace.home is None:
                self._save_note = "demo — not saved"
                return
            self.workspace.save()
            self._save_note = f"saved {datetime.now().strftime('%H:%M:%S')}"

        def action_drill_down(self, series: str) -> None:
            """Open the forecast screen focused on ``series``.

            Switches to the forecast screen and selects the matching series in
            its existing series list; the selection is applied on a later
            refresh cycle so it lands after the list has been (re)built.
            """
            self._drill_target = str(series)
            self.switch_screen("forecast")
            self.call_after_refresh(self._apply_drill_target)

        def _apply_drill_target(self, _attempts: int = 0) -> None:
            target = self._drill_target
            if target is None:
                return
            listview = None
            if type(self.screen) is ForecastScreen:
                try:
                    listview = self.screen.query_one("#series-list", ListView)
                except Exception:
                    listview = None
            if listview is not None and len(listview.children):
                for i, item in enumerate(listview.children):
                    if getattr(item, "name", None) == target:
                        listview.index = i
                        break
                self._drill_target = None
                return
            # The screen may not be mounted / the list not built yet; retry a
            # bounded number of times before giving up so we never hot-loop.
            if _attempts < 12:
                self.call_after_refresh(self._apply_drill_target, _attempts + 1)
            else:
                self._drill_target = None

        # -- workers -----------------------------------------------------

        @work(thread=True, exclusive=True, group="refresh")
        def _refresh_worker(self) -> None:
            # exclusive=True cannot cancel a thread worker mid-run, so two
            # overlapping refreshes both finish and the slower one can land
            # last. Stamp the result with the generation this worker started
            # on so a superseded refresh can be dropped instead of reinstating
            # its older panel.
            generation = self._generation
            settings = self.workspace.settings
            alerts = self.workspace.alerts
            try:
                panel = self.provider.refresh()
                rows = engine_bridge.dashboard_rows(panel, settings)
                forecasts = rows.rename(columns={"next": "yhat"})[["unique_id", "yhat"]]
                governance = None
                if any(a.get("kind") == "coverage_below" for a in alerts):
                    governance = engine_bridge.governance_frame(panel, settings)
                fired = engine_bridge.alerts_fired(
                    panel, forecasts, alerts, governance=governance
                )
            except Exception as exc:
                self.call_from_thread(self._refresh_failed, exc, generation)
                return
            self.call_from_thread(
                self._apply_refresh, panel, rows, fired, governance, generation
            )

        def _apply_refresh(
            self, panel, rows, fired, governance, generation: int | None = None
        ) -> None:
            if generation is not None and generation != self._generation:
                # a newer refresh superseded this one: installing this panel
                # would roll the console back to older data (and its
                # governance frame past the _apply_governance guard). Leave
                # `refreshing` set — the newer worker clears it when it lands.
                return
            self.panel = panel
            self.rows = rows
            self.fired_alerts = fired
            if governance is not None:
                self.governance = governance
            self.last_refresh = datetime.now().strftime("%H:%M:%S")
            self.refreshing = False
            self.alerts_count = len(fired)
            self._update_status()
            self._update_active_screen()

        def _refresh_failed(self, exc: Exception, generation: int | None = None) -> None:
            if generation is not None and generation != self._generation:
                # a superseded refresh failing must not clobber the alerts a
                # newer, successful refresh already published
                return
            self.fired_alerts = [f"refresh failed: {exc}"]
            self.refreshing = False
            self.alerts_count = len(self.fired_alerts)
            self._update_status()
            self._update_active_screen()

        def request_board(self) -> None:
            """Compute the leaderboard in a worker.

            A no-op while one is already running, and while a refresh is in
            flight: ``self.panel`` is still the pre-refresh one then, so the
            board would be computed from stale data yet stamped with the
            post-refresh generation, which the ``_apply_board`` guard cannot
            detect. ``_apply_refresh`` re-renders the active screen, and the
            screen re-requests because ``board`` is still ``None``.
            """
            if (
                self.board is None
                and self.panel is not None
                and not self.refreshing
                and not self._board_working
            ):
                self._board_working = True
                self._board_worker()

        @work(thread=True, exclusive=True, group="board")
        def _board_worker(self) -> None:
            generation = self._generation
            try:
                board = engine_bridge.leaderboard_frame(self.panel, self.workspace.settings)
            except Exception as exc:
                board = ComputeFailure(str(exc))
            self.call_from_thread(self._apply_board, board, generation)

        def _apply_board(self, board, generation: int | None = None) -> None:
            self._board_working = False
            if generation is not None and generation != self._generation:
                # a refresh landed while this worker ran: its board describes the
                # pre-refresh panel. Leave self.board at None and let the active
                # screen re-request, rather than resurrecting a stale board that
                # request_board() would then never recompute.
                self._update_active_screen()
                return
            self.board = board  # DataFrame or ComputeFailure — never back to None
            self._update_active_screen()

        def request_governance(self) -> None:
            """Compute the governance frame in a worker.

            A no-op while one is already running or a refresh is in flight —
            see :meth:`request_board` for why the in-flight case matters.
            """
            if (
                self.governance is None
                and self.panel is not None
                and not self.refreshing
                and not self._governance_working
            ):
                self._governance_working = True
                self._governance_worker()

        @work(thread=True, exclusive=True, group="governance")
        def _governance_worker(self) -> None:
            generation = self._generation
            try:
                governance = engine_bridge.governance_frame(
                    self.panel, self.workspace.settings
                )
            except Exception as exc:
                governance = ComputeFailure(str(exc))
            self.call_from_thread(self._apply_governance, governance, generation)

        def _apply_governance(self, governance, generation: int | None = None) -> None:
            self._governance_working = False
            if generation is not None and generation != self._generation:
                # stale: computed from the pre-refresh panel (see _apply_board)
                self._update_active_screen()
                return
            self.governance = governance  # DataFrame or ComputeFailure, not None
            self._update_active_screen()

        # -- status line -------------------------------------------------

        def watch_refreshing(self, _: bool) -> None:
            self._update_status()

        def watch_alerts_count(self, _: int) -> None:
            self._update_status()

        def _update_status(self) -> None:
            """The header sub-title: home · panel shape · refresh time · alerts."""
            if self.panel is None:
                shape = "—"
            else:
                shape = f"{len(self.panel)}r×{self.panel['unique_id'].nunique()}s"
            parts = [
                f"home {self.workspace.home or '—'}",
                f"panel {shape}",
                f"refreshed {self.last_refresh or 'never'}",
            ]
            if self.refreshing:
                parts.append("refreshing…")
            if self.alerts_count:
                parts.append(f"⚠ {self.alerts_count} alert(s)")
            if self._save_note:
                parts.append(self._save_note)
            self.sub_title = " · ".join(parts)

        def _update_active_screen(self) -> None:
            update = getattr(self.screen, "update_view", None)
            if update is None:
                return
            try:
                update()
            except Exception:
                # A worker callback can land mid screen-switch or shutdown, when
                # the screen's widgets are gone (query_one -> NoMatches). A view
                # refresh failing must never kill the worker or the app; the
                # screen re-renders itself on its next on_screen_resume.
                pass

else:  # textual not installed: keep the module importable
    ForecastOSApp = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forecast-os-tui",
        description="The always-on forecast-os terminal console.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run on the built-in seeded demo panel (no workspace reads/writes)",
    )
    parser.add_argument(
        "--data",
        action="append",
        metavar="PATH",
        help="CSV source to load (repeatable)",
    )
    parser.add_argument(
        "--mapping",
        metavar="NAME",
        help="registered schema mapping applied to every --data source",
    )
    parser.add_argument(
        "--home",
        metavar="PATH",
        help="workspace directory (default $FORECAST_OS_HOME or ~/.forecast-os)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Console entry point for ``forecast-os-tui``.

    Raises ``ImportError`` with the ``forecast-os[terminal]`` install hint
    when the optional textual dependency is missing.
    """
    args = _build_parser().parse_args(argv)
    if not _HAS_TEXTUAL:
        raise ImportError(_TERMINAL_HINT)
    if args.demo:
        workspace = Workspace(home=Workspace.resolve_home(args.home))
    else:
        try:
            workspace = Workspace.load(home=args.home)
        except ForecastOSError as exc:
            raise SystemExit(f"forecast-os-tui: {exc}") from exc
        for path in args.data or []:
            workspace.sources.append(
                {"path": path, "mapping": args.mapping, "overrides": {}}
            )
    ForecastOSApp(workspace, demo=args.demo).run()


if __name__ == "__main__":  # pragma: no cover
    main()
