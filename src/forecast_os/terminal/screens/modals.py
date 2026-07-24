"""Modal forms for editing workspace sources and alerts.

Two :class:`textual.screen.ModalScreen` subclasses that collect one record
each and return it to their caller through ``dismiss()``:

- :class:`SourceModal` edits a data source (``path``, an optional registered
  schema ``mapping``, and a ``freq`` override) and dismisses the
  ``{"path", "mapping", "overrides"}`` dict the workspace stores (``freq``
  lands in ``overrides['freq']`` when given). Cancelling dismisses ``None``.
- :class:`AlertModal` edits an alert rule (``kind``, ``series``, numeric
  ``threshold``) and dismisses the ``{"kind", "series", "threshold"}`` dict.
  A threshold that does not parse as a float keeps the form open.

The ``textual`` dependency is the optional ``[terminal]`` extra: importing
this module always succeeds (mirroring :mod:`forecast_os.terminal.app`), and
the :func:`new_source_modal` / :func:`new_alert_modal` factories raise
``ImportError`` with the install hint when the extra is missing.
"""

from __future__ import annotations

from ...connectors.base import list_mappings

_TERMINAL_HINT = (
    "the terminal UI needs the optional textual dependency; "
    'install it with: pip install "forecast-os[terminal]"'
)

try:
    from textual.app import ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.screen import ModalScreen
    from textual.widgets import Button, Input, Label, Select, Static

    _HAS_TEXTUAL = True
except ImportError:  # tests exercise this path by monkeypatching the flag
    _HAS_TEXTUAL = False

__all__ = ["SourceModal", "AlertModal", "new_source_modal", "new_alert_modal"]


_MODAL_CSS = """
SourceModal, AlertModal {
    align: center middle;
}
#dialog {
    width: 64;
    max-width: 90%;
    height: auto;
    padding: 1 2;
    border: thick $primary;
    background: $surface;
}
#dialog-title {
    text-style: bold;
    padding-bottom: 1;
}
.dialog-error {
    color: $error;
    height: auto;
}
#dialog-buttons {
    height: auto;
    padding-top: 1;
    align-horizontal: right;
}
#dialog-buttons Button {
    margin-left: 2;
}
"""


if _HAS_TEXTUAL:

    class SourceModal(ModalScreen[dict | None]):
        """Add or edit one workspace data source; dismiss the source dict."""

        CSS = _MODAL_CSS
        BINDINGS = [("escape", "cancel", "Cancel")]

        def __init__(self, source: dict | None = None):
            super().__init__()
            self.source = source

        def compose(self) -> ComposeResult:
            src = self.source or {}
            overrides = src.get("overrides") or {}
            with Vertical(id="dialog"):
                yield Label(
                    "Edit source" if self.source else "Add source", id="dialog-title"
                )
                yield Label("path")
                yield Input(
                    value=str(src.get("path") or ""),
                    placeholder="path/to/data.csv",
                    id="source-path",
                )
                yield Label("mapping")
                yield Select(
                    self._mapping_options(),
                    value=self._initial_mapping(),
                    allow_blank=True,
                    prompt="— (contract columns)",
                    id="source-mapping",
                )
                yield Label("freq override")
                yield Input(
                    value=str(overrides.get("freq") or ""),
                    placeholder="e.g. MS, W, D — blank to leave unset",
                    id="source-freq",
                )
                yield Static("", id="source-error", classes="dialog-error")
                with Horizontal(id="dialog-buttons"):
                    yield Button("Save", variant="primary", id="source-save")
                    yield Button("Cancel", id="source-cancel")

        @staticmethod
        def _mapping_options() -> list[tuple[str, str]]:
            return [(name, name) for name in list_mappings()["name"].tolist()]

        def _initial_mapping(self):
            current = (self.source or {}).get("mapping")
            names = list_mappings()["name"].tolist()
            return current if current in names else Select.NULL

        def action_submit(self) -> None:
            """Build the source dict from the fields and dismiss it."""
            path = self.query_one("#source-path", Input).value.strip()
            if not path:
                self.query_one("#source-error", Static).update("path is required")
                return
            mapping = self.query_one("#source-mapping", Select).value
            mapping = None if mapping is Select.NULL else str(mapping)
            freq = self.query_one("#source-freq", Input).value.strip()
            overrides = dict((self.source or {}).get("overrides") or {})
            overrides.pop("freq", None)
            if freq:
                overrides["freq"] = freq
            self.dismiss({"path": path, "mapping": mapping, "overrides": overrides})

        def action_cancel(self) -> None:
            self.dismiss(None)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "source-save":
                self.action_submit()
            else:
                self.action_cancel()

    class AlertModal(ModalScreen[dict | None]):
        """Add or edit one alert rule; dismiss the alert dict."""

        CSS = _MODAL_CSS
        BINDINGS = [("escape", "cancel", "Cancel")]

        _KINDS = [
            ("forecast_below", "forecast_below"),
            ("coverage_below", "coverage_below"),
        ]

        def __init__(self, alert: dict | None = None):
            super().__init__()
            self.alert = alert

        def compose(self) -> ComposeResult:
            a = self.alert or {}
            threshold = a.get("threshold")
            with Vertical(id="dialog"):
                yield Label(
                    "Edit alert" if self.alert else "Add alert", id="dialog-title"
                )
                yield Label("kind")
                yield Select(
                    self._KINDS,
                    value=a.get("kind") or "forecast_below",
                    allow_blank=False,
                    id="alert-kind",
                )
                yield Label("series ('*' for all)")
                yield Input(
                    value=str(a.get("series") or "*"),
                    placeholder="* or a unique_id",
                    id="alert-series",
                )
                yield Label("threshold")
                yield Input(
                    value="" if threshold is None else str(threshold),
                    placeholder="e.g. 100",
                    id="alert-threshold",
                )
                yield Static("", id="alert-error", classes="dialog-error")
                with Horizontal(id="dialog-buttons"):
                    yield Button("Save", variant="primary", id="alert-save")
                    yield Button("Cancel", id="alert-cancel")

        def action_submit(self) -> None:
            """Validate the threshold parses, then dismiss the alert dict."""
            kind = self.query_one("#alert-kind", Select).value
            if kind is Select.NULL:
                kind = "forecast_below"
            series = self.query_one("#alert-series", Input).value.strip() or "*"
            raw = self.query_one("#alert-threshold", Input).value.strip()
            try:
                threshold = float(raw)
            except ValueError:
                self.query_one("#alert-error", Static).update(
                    "threshold must be a number"
                )
                return
            self.dismiss({"kind": str(kind), "series": series, "threshold": threshold})

        def action_cancel(self) -> None:
            self.dismiss(None)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "alert-save":
                self.action_submit()
            else:
                self.action_cancel()

else:  # textual not installed: keep the module importable
    SourceModal = None
    AlertModal = None


def new_source_modal(source: dict | None = None):
    """Construct a :class:`SourceModal` (needs the ``[terminal]`` extra).

    Raises ``ImportError`` with the ``forecast-os[terminal]`` install hint
    when textual is not installed.
    """
    if not _HAS_TEXTUAL:
        raise ImportError(_TERMINAL_HINT)
    return SourceModal(source=source)


def new_alert_modal(alert: dict | None = None):
    """Construct an :class:`AlertModal` (needs the ``[terminal]`` extra).

    Raises ``ImportError`` with the ``forecast-os[terminal]`` install hint
    when textual is not installed.
    """
    if not _HAS_TEXTUAL:
        raise ImportError(_TERMINAL_HINT)
    return AlertModal(alert=alert)
