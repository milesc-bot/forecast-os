"""The leaderboard screen: models ranked on the current panel.

Renders :func:`~forecast_os.terminal.engine_bridge.leaderboard_frame`
(already sorted best-first by mase) in a table; the board is computed in an
app worker thread on first visit and cached until the next refresh.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from ..engine_bridge import ComputeFailure


class LeaderboardScreen(Screen):
    """Model comparison table: mase + interval coverage, best model first."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="leaderboard-table")
        yield Static(
            "[dim]sorted by mase (lower is better) · coverage should sit near "
            "level/100 · r recomputes[/]",
            id="leaderboard-hint",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#leaderboard-table", DataTable).cursor_type = "row"
        self.update_view()

    def on_screen_resume(self) -> None:
        self.update_view()

    def update_view(self) -> None:
        """Render the cached board, or ask the app to compute one."""
        app = self.app
        table = self.query_one("#leaderboard-table", DataTable)
        board = getattr(app, "board", None)
        table.clear(columns=True)
        if isinstance(board, ComputeFailure):
            table.add_column("model")
            table.add_row(f"[red]leaderboard unavailable: {board.reason}[/]")
            return
        if board is None:
            table.add_column("model")
            table.add_row("[dim]computing leaderboard…[/]")
            app.request_board()
            return
        table.add_columns(*[str(c) for c in board.columns])
        for row in board.itertuples(index=False):
            cells = [
                value if isinstance(value, str) else f"{value:.3f}" for value in row
            ]
            table.add_row(*cells)
