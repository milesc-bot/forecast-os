"""The forecast screen: a per-series fan chart with keyboard series switching.

A series list on the left, a plotext chart on the right: the history line,
the forecast line, and the prediction-interval band drawn as secondary
lo/hi lines, all from
:func:`~forecast_os.terminal.engine_bridge.forecast_frame`. ``n``/``p``
(or the list's arrow keys) switch series.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, ListItem, ListView
from textual_plotext import PlotextPlot

from .. import engine_bridge


class ForecastScreen(Screen):
    """Series list + fan chart for the workspace model/h/level settings."""

    BINDINGS = [
        ("n", "next_series", "Next series"),
        ("p", "prev_series", "Prev series"),
    ]

    #: Series currently shown in the list (update can arrive before on_mount).
    _series: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="forecast-body"):
            yield ListView(id="series-list")
            yield PlotextPlot(id="fan-chart")
        yield Footer()

    def on_mount(self) -> None:
        self.update_view()

    def on_screen_resume(self) -> None:
        self.update_view()

    def update_view(self) -> None:
        """Sync the series list with the panel and plot the highlighted series."""
        panel = getattr(self.app, "panel", None)
        chart = self.query_one("#fan-chart", PlotextPlot)
        if panel is None:
            chart.plt.clear_data()
            chart.plt.title("waiting for data — press r to refresh")
            chart.refresh()
            return
        series = sorted(panel["unique_id"].astype(str).unique())
        if series != self._series:
            self._series = series
            series_list = self.query_one("#series-list", ListView)
            series_list.clear()
            series_list.extend(ListItem(Label(uid), name=uid) for uid in series)
            series_list.index = 0
        self._plot_current()

    def _plot_current(self) -> None:
        series_list = self.query_one("#series-list", ListView)
        item = series_list.highlighted_child
        if item is not None and item.name:
            self._plot(item.name)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is not None and event.item.name:
            self._plot(event.item.name)

    def action_next_series(self) -> None:
        self._step_series(1)

    def action_prev_series(self) -> None:
        self._step_series(-1)

    def _step_series(self, step: int) -> None:
        series_list = self.query_one("#series-list", ListView)
        count = len(series_list.children)
        if count:
            series_list.index = ((series_list.index or 0) + step) % count

    def _plot(self, series: str) -> None:
        """Fan chart for ``series``: history, forecast, and interval band lines."""
        app = self.app
        panel = getattr(app, "panel", None)
        if panel is None:
            return
        chart = self.query_one("#fan-chart", PlotextPlot)
        plt = chart.plt
        plt.clear_data()
        settings = app.workspace.settings
        try:
            history, forecast = engine_bridge.forecast_frame(panel, series, settings)
        except Exception as exc:
            plt.title(f"{series}: {exc}")
            chart.refresh()
            return
        n = len(history)
        x_hist = list(range(n))
        y_hist = history["y"].tolist()
        # prepend the last actual so the forecast lines connect to history
        x_fc = list(range(n - 1, n + len(forecast)))
        last = y_hist[-1]
        y_fc = [last, *forecast["yhat"].tolist()]
        lo = [last, *forecast["lo"].tolist()]
        hi = [last, *forecast["hi"].tolist()]
        plt.plot(x_hist, y_hist, label="history")
        plt.plot(x_fc, y_fc, label="forecast")
        plt.plot(x_fc, lo, label="lo", marker="dot")
        plt.plot(x_fc, hi, label="hi", marker="dot")
        labels = history["ds"].dt.strftime("%y-%m")
        fc_end = forecast["ds"].iloc[-1]
        ticks = [0, n - 1, x_fc[-1]]
        plt.xticks(ticks, [labels.iloc[0], labels.iloc[-1], fc_end.strftime("%y-%m")])
        plt.title(
            f"{series} · {settings.get('model')} h={settings.get('h')} "
            f"level={settings.get('level')}"
        )
        chart.refresh()
