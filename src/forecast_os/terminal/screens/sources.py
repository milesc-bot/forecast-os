"""The sources screen: interactive source and alert editing.

Add, edit, and remove workspace data sources (``a`` / ``e`` / ``x``) and add
or delete forecast alerts (``A`` / ``X``) through modal forms, alongside the
read-only catalog of registered schema mappings. Every mutation is persisted
through the app (:meth:`~forecast_os.terminal.app.ForecastOSApp.save_workspace`,
a no-op in ``--demo``) and triggers a refresh so the rest of the console
reflects the change.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from ...connectors.base import list_mappings
from .modals import new_alert_modal, new_source_modal


class SourcesScreen(Screen):
    """Editable workspace sources and alerts plus the mapping catalog."""

    BINDINGS = [
        ("a", "add_source", "Add source"),
        ("e", "edit_source", "Edit source"),
        ("x", "delete_source", "Delete source"),
        ("A", "add_alert", "Add alert"),
        ("X", "delete_alert", "Delete alert"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            "[b]workspace sources[/] — [dim]a add · e edit · x delete[/]",
            id="sources-title",
        )
        yield DataTable(id="sources-table")
        yield Static(
            "[b]alerts[/] — [dim]A add · X delete[/]",
            id="alerts-title",
        )
        yield DataTable(id="alerts-table")
        yield Static("[b]registered mappings[/]", id="mappings-title")
        yield DataTable(id="mappings-table")
        yield Static(
            "[dim]edits persist to workspace.json (a no-op in --demo)[/]",
            id="sources-hint",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#sources-table", DataTable).cursor_type = "row"
        self.query_one("#alerts-table", DataTable).cursor_type = "row"
        mappings = self.query_one("#mappings-table", DataTable)
        mappings.add_columns("name", "description", "freq", "agg")
        for row in list_mappings().itertuples(index=False):
            mappings.add_row(row.name, row.description, row.freq, row.agg)
        self.update_view()

    def on_screen_resume(self) -> None:
        self.update_view()

    def update_view(self) -> None:
        """Re-render the source and alert tables from the workspace."""
        self._render_sources()
        self._render_alerts()

    def _render_sources(self) -> None:
        table = self.query_one("#sources-table", DataTable)
        table.clear()
        if not table.columns:  # update can arrive before on_mount runs
            table.add_columns("path", "mapping", "overrides")
        sources = self.app.workspace.sources
        if not sources:
            table.add_row("[dim]none — showing the built-in demo panel[/]", "—", "—")
            return
        for src in sources:
            overrides = src.get("overrides") or {}
            table.add_row(
                str(src.get("path")),
                str(src.get("mapping") or "— (contract columns)"),
                ", ".join(f"{k}={v}" for k, v in overrides.items()) or "—",
            )

    def _render_alerts(self) -> None:
        table = self.query_one("#alerts-table", DataTable)
        table.clear()
        if not table.columns:  # update can arrive before on_mount runs
            table.add_columns("kind", "series", "threshold")
        alerts = self.app.workspace.alerts
        if not alerts:
            table.add_row("[dim]none — press A to add an alert[/]", "—", "—")
            return
        for alert in alerts:
            table.add_row(
                str(alert.get("kind")),
                str(alert.get("series")),
                str(alert.get("threshold")),
            )

    # -- persistence -----------------------------------------------------

    def _persist_and_refresh(self) -> None:
        """Save the workspace (no-op in demo), re-render, and recompute."""
        self.app.save_workspace()
        self.update_view()
        self.app.action_refresh()

    def _selected_index(self, table_id: str, items: list) -> int | None:
        """The highlighted row index, or ``None`` when nothing is selectable."""
        if not items:
            return None
        row = self.query_one(table_id, DataTable).cursor_row
        if row is None or not (0 <= row < len(items)):
            return None
        return row

    # -- source actions --------------------------------------------------

    def action_add_source(self) -> None:
        def on_result(result: dict | None) -> None:
            if result:
                self.app.workspace.sources.append(result)
                self._persist_and_refresh()

        self.app.push_screen(new_source_modal(), on_result)

    def action_edit_source(self) -> None:
        sources = self.app.workspace.sources
        idx = self._selected_index("#sources-table", sources)
        if idx is None:
            return

        def on_result(result: dict | None) -> None:
            if result:
                self.app.workspace.sources[idx] = result
                self._persist_and_refresh()

        self.app.push_screen(new_source_modal(source=sources[idx]), on_result)

    def action_delete_source(self) -> None:
        sources = self.app.workspace.sources
        idx = self._selected_index("#sources-table", sources)
        if idx is None:
            return
        del sources[idx]
        self._persist_and_refresh()

    # -- alert actions ---------------------------------------------------

    def action_add_alert(self) -> None:
        def on_result(result: dict | None) -> None:
            if result:
                self.app.workspace.alerts.append(result)
                self._persist_and_refresh()

        self.app.push_screen(new_alert_modal(), on_result)

    def action_delete_alert(self) -> None:
        alerts = self.app.workspace.alerts
        idx = self._selected_index("#alerts-table", alerts)
        if idx is None:
            return
        del alerts[idx]
        self._persist_and_refresh()
