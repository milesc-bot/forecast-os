"""The dashboard screen: the watchlist at a glance plus fired alerts.

One row per watched series — last actual, fast 1-step outlook, percent
delta, and a 12-period sparkline — rendered from
:func:`~forecast_os.terminal.engine_bridge.dashboard_rows`, with the fired
alert messages beneath the table. An empty workspace watchlist shows every
series.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static


class DashboardScreen(Screen):
    """Watchlist table of per-series outlooks with an alerts panel below."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="dashboard-table")
        yield Static(id="dashboard-alerts")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#dashboard-table", DataTable).cursor_type = "row"
        self.update_view()

    def on_screen_resume(self) -> None:
        self.update_view()

    def update_view(self) -> None:
        """Re-render the table and alert lines from the app's computed state."""
        app = self.app
        table = self.query_one("#dashboard-table", DataTable)
        alerts = self.query_one("#dashboard-alerts", Static)
        rows = getattr(app, "rows", None)
        table.clear()
        if not table.columns:  # update can arrive before on_mount runs
            table.add_columns("series", "last", "next", "Δ%", "last 12")

        # Fired alerts render first, so a failed first refresh (e.g. a bad
        # source) shows its reason even before any rows exist.
        fired = getattr(app, "fired_alerts", [])
        if fired:
            alerts.update("\n".join(f"[bold red]! {message}[/]" for message in fired))
        elif rows is None:
            hint = "refreshing…" if getattr(app, "refreshing", False) else "press r to refresh"
            alerts.update(f"[dim]{hint}[/]")
        else:
            alerts.update("[dim]no alerts fired[/]")

        if rows is None:
            return
        watch = app.workspace.watch
        view = rows if not watch else rows[rows["unique_id"].isin(watch)]
        for row in view.itertuples(index=False):
            table.add_row(
                row.unique_id,
                f"{row.last:,.1f}",
                f"{row.next:,.1f}",
                f"{row.delta_pct:+.1f}%" if row.delta_pct == row.delta_pct else "—",
                row.spark,
            )
