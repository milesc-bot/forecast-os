"""The terminal UI screens: dashboard, forecast, leaderboard, governance, sources.

Each module defines one :class:`textual.screen.Screen` that renders a frame
from :mod:`forecast_os.terminal.engine_bridge`. The modules import textual at
the top level, so they are only imported from the guarded block in
:mod:`forecast_os.terminal.app` — this package's ``__init__`` deliberately
imports nothing, keeping ``forecast_os.terminal`` importable without the
optional ``[terminal]`` extra.
"""
