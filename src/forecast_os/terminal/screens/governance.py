"""The governance screen: per-cutoff accuracy, bias, and coverage.

Renders :func:`~forecast_os.terminal.engine_bridge.governance_frame` with
review-worthy cells highlighted: ``pct_bias`` below -5% (sandbagging — the
forecast systematically ran low) in red, and coverage more than 15 points
under the nominal level in yellow.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from ..engine_bridge import ComputeFailure

#: pct_bias below this fraction is flagged as sandbagging.
SANDBAG_THRESHOLD = -0.05

#: Coverage this far below the nominal level/100 is flagged.
COVERAGE_SLACK = 0.15


class GovernanceScreen(Screen):
    """Per-(series, cutoff) mase / pct_bias / coverage table with flags."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="governance-table")
        yield Static(
            "[dim]red pct_bias = sandbagging (< -5%) · yellow coverage = "
            "intervals running narrow · r recomputes[/]",
            id="governance-hint",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#governance-table", DataTable).cursor_type = "row"
        self.update_view()

    def on_screen_resume(self) -> None:
        self.update_view()

    def update_view(self) -> None:
        """Render the cached governance frame, or ask the app to compute one."""
        app = self.app
        table = self.query_one("#governance-table", DataTable)
        governance = getattr(app, "governance", None)
        table.clear()
        if not table.columns:  # update can arrive before on_mount runs
            table.add_columns("series", "cutoff", "mase", "pct_bias", "coverage")
        if isinstance(governance, ComputeFailure):
            table.add_row(
                f"[red]governance unavailable: {governance.reason}[/]", "", "", "", ""
            )
            return
        if governance is None:
            table.add_row("[dim]computing governance table…[/]", "", "", "", "")
            app.request_governance()
            return
        nominal = int(app.workspace.settings.get("level", 80)) / 100.0
        for row in governance.itertuples(index=False):
            cutoff = row.cutoff
            cutoff_label = (
                cutoff.strftime("%Y-%m-%d") if hasattr(cutoff, "strftime") else str(cutoff)
            )
            bias_cell = Text(f"{row.pct_bias:+.3f}")
            if row.pct_bias < SANDBAG_THRESHOLD:
                bias_cell.stylize("bold red")
            coverage_cell = Text(f"{row.coverage:.2f}")
            if row.coverage < nominal - COVERAGE_SLACK:
                coverage_cell.stylize("bold yellow")
            table.add_row(
                row.unique_id, cutoff_label, f"{row.mase:.3f}", bias_cell, coverage_cell
            )
