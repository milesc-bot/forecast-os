# The terminal

`forecast-os-tui` is an always-on, keyboard-driven console over the engine
(extra: `pip install "forecast-os[terminal]"`).

```bash
forecast-os-tui --demo                                    # seeded GTM console
forecast-os-tui --data deals.csv --mapping hubspot_deals  # your data
forecast-os-tui --home ~/team-workspace                   # alternate workspace
```

## Screens & keys

| Key | Screen | What it shows |
|---|---|---|
| `d` | Dashboard | watchlist: last value, next-period forecast, Δ%, sparkline, fired alerts |
| `f` | Forecast | fan chart (history + forecast + interval band) per series, `n`/`p` to switch |
| `l` | Leaderboard | model comparison (MASE + coverage) on the loaded panel |
| `g` | Governance | per-cutoff MASE, signed bias (sandbagging in red), coverage |
| `s` | Sources | configured sources and the mapping catalog |
| `r` | — | refresh (background worker; UI never blocks) |
| `q` | — | quit |

## Workspace

State persists in `~/.forecast-os/workspace.json` (`$FORECAST_OS_HOME`
overrides): data sources (path + mapping + overrides), a watchlist, settings
(`model`, `h`, `level`, `season_length`, `refresh_seconds` — set > 0 for
auto-refresh), and alert rules:

```json
{
  "sources": [{"path": "deals.csv", "mapping": "hubspot_deals", "overrides": {}}],
  "settings": {"model": "auto_ets", "h": 6, "level": 80, "refresh_seconds": 300},
  "alerts": [
    {"kind": "forecast_below", "series": "total", "threshold": 1000000},
    {"kind": "coverage_below", "series": "*", "threshold": 0.6}
  ]
}
```

`--demo` never touches the workspace — it boots on seeded GTM data so the
first run is alive immediately.

## Scaffold status

This is the v0.5.0 scaffold: read-only sources screen (editing forms are the
next step), no snapshot store yet, single-panel workspaces. The compute layer
(`forecast_os.terminal.engine_bridge`) is pure functions over the public
engine API, so screens stay thin and testable.
