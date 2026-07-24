"""The always-on terminal UI for forecast-os.

A keyboard-first console built on the optional ``textual`` dependency
(``pip install "forecast-os[terminal]"``, console script ``forecast-os-tui``).
The package splits into textual-free layers that are unit-testable anywhere —
:mod:`.workspace` (persisted configuration), :mod:`.data` (panel loading),
:mod:`.engine_bridge` (the compute functions the screens render) — and the
guarded UI in :mod:`.app` / :mod:`.screens`, which import textual only when
it is installed.
"""
