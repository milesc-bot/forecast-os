"""Panel loading for the terminal UI: workspace sources in, contract panel out.

:class:`PanelProvider` turns a :class:`~forecast_os.terminal.workspace.Workspace`
into a validated ``(unique_id, ds, y)`` panel — each source CSV is read through
the connectors layer (with its schema mapping applied when one is configured)
and the results are concatenated. A workspace with no sources falls back to
:func:`demo_panel`, a seeded GTM-style bookings panel, so a first launch is
never blank. No textual imports here; everything is testable headless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..connectors.files import CSVSource
from ..core.exceptions import ForecastOSError
from ..core.types import ID_COL, REQUIRED_COLS, TARGET_COL, TIME_COL, validate_panel
from .workspace import Workspace

__all__ = ["PanelProvider", "demo_panel"]

#: The demo team: (team, rep, base monthly bookings in $k).
_DEMO_REPS = (
    ("west", "alice", 95.0),
    ("west", "bob", 70.0),
    ("west", "cara", 55.0),
    ("east", "dan", 85.0),
    ("east", "erin", 62.0),
    ("east", "fay", 48.0),
)


def demo_panel(months: int = 30, seed: int = 7) -> pd.DataFrame:
    """A seeded GTM demo panel: 6 reps, monthly bookings with quarter-end pushes.

    Six ``team/rep`` series (``west/alice`` ... ``east/fay``) over ``months``
    monthly periods, each with its own base level ($k scale), a gentle upward
    trend, a quarter-end seasonal bump, and light noise — the shape of a CRM
    bookings export after :func:`~forecast_os.gtm.to_panel`. Deterministic for
    a given ``seed`` and small enough to fit in well under a second.
    """
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2024-01-01", periods=months, freq="MS")
    quarter_end = (ds.month % 3 == 0).astype(float)
    t = np.arange(months, dtype=float)
    frames = []
    for team, rep, base in _DEMO_REPS:
        trend = base * rng.uniform(0.004, 0.012)
        amp = base * rng.uniform(0.15, 0.30)
        noise = base * 0.04 * rng.standard_normal(months)
        y = np.maximum(base + trend * t + amp * quarter_end + noise, 0.0)
        frames.append(
            pd.DataFrame({ID_COL: f"{team}/{rep}", TIME_COL: ds, TARGET_COL: y})
        )
    return validate_panel(pd.concat(frames, ignore_index=True))


class PanelProvider:
    """Load the workspace's sources into one validated contract panel.

    Each source is a CSV read via the connectors layer: with a ``mapping``
    configured, :meth:`~forecast_os.connectors.base.Source.to_panel` applies
    the schema recipe (plus the source's ``overrides``); without one, the file
    must already carry the contract columns ``(unique_id, ds, y)``. A failing
    source raises :class:`ForecastOSError` naming the source and the reason.
    An empty source list yields :func:`demo_panel`.
    """

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def load(self) -> pd.DataFrame:
        """Read every source and return the concatenated, validated panel."""
        sources = self.workspace.sources
        if not sources:
            return demo_panel()
        frames = []
        for src in sources:
            path = src.get("path")
            mapping = src.get("mapping")
            overrides = dict(src.get("overrides") or {})
            try:
                source = CSVSource(path, mapping=mapping)
                if mapping:
                    frame = source.to_panel(**overrides)
                else:
                    records = source.fetch()
                    missing = [c for c in REQUIRED_COLS if c not in records.columns]
                    if missing:
                        raise ForecastOSError(
                            f"missing contract column(s) {missing}; files without "
                            f"a mapping must provide (unique_id, ds, y)"
                        )
                    frame = validate_panel(records[list(REQUIRED_COLS)])
            except Exception as exc:
                raise ForecastOSError(
                    f"source {str(path)!r} failed to load: {exc}"
                ) from exc
            frames.append(frame)
        try:
            return validate_panel(pd.concat(frames, ignore_index=True))
        except ForecastOSError as exc:
            raise ForecastOSError(
                f"combined panel from {len(frames)} source(s) is invalid: {exc}"
            ) from exc

    def refresh(self) -> pd.DataFrame:
        """Re-read every source (sources are cheap file reads); alias of :meth:`load`."""
        return self.load()
