"""The sources screen: what the panel is built from (read-only scaffold).

Shows the workspace's configured sources and the registry of schema
mappings available to them. This is display-only for now — add/edit/remove
forms are the next step; today sources are edited in ``workspace.json`` or
passed with ``--data``/``--mapping``.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from ...connectors.base import list_mappings


class SourcesScreen(Screen):
    """Configured workspace sources plus the registered mapping catalog."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("[b]workspace sources[/]", id="sources-title")
        yield DataTable(id="sources-table")
        yield Static("[b]registered mappings[/]", id="mappings-title")
        yield DataTable(id="mappings-table")
        yield Static(
            "[dim]read-only scaffold — editing forms are a next step; configure "
            "sources in workspace.json or with --data/--mapping[/]",
            id="sources-hint",
        )
        yield Footer()

    def on_mount(self) -> None:
        mappings = self.query_one("#mappings-table", DataTable)
        mappings.add_columns("name", "description", "freq", "agg")
        for row in list_mappings().itertuples(index=False):
            mappings.add_row(row.name, row.description, row.freq, row.agg)
        self.update_view()

    def on_screen_resume(self) -> None:
        self.update_view()

    def update_view(self) -> None:
        """Re-render the source list from the workspace."""
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
