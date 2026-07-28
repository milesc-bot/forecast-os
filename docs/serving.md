# Serving: HTTP and MCP

The same engine reaches agents and services through two front doors that
share one set of tool functions, so a forecast returns identically whether it
came over HTTP or through an MCP client.

## REST API

`pip install "forecast-os-gtm[serve]"` then:

```bash
forecast-os-serve --host 0.0.0.0 --port 8000
```

| Method & path | Body | Returns |
|---|---|---|
| `GET /health` | — | `{"status": "ok", "version": ...}` |
| `GET /models` | — | the model catalog |
| `GET /mappings` | — | the connector recipe catalog |
| `POST /preview` | `{records, mapping?, freq?, agg?}` | shaped-panel preview |
| `POST /forecast` | `{records, mapping?, model?, h?, level?, ...}` | `{"forecast": [...]}` |
| `POST /compare` | `{records, models?, h?, metrics?, level?, ...}` | leaderboard + failures |
| `POST /quota` | `{records, quota, h?, level?, ...}` | attainment + unmatched keys |

```bash
curl -s localhost:8000/forecast -H 'content-type: application/json' -d '{
  "records": [{"unique_id":"total","ds":"2026-01-01","y":100}, ...],
  "model": "auto_ets", "h": 6, "level": [80]
}'
```

Contract violations return HTTP 400 with `{"error": "..."}` — never a
traceback. Handlers are thin wrappers over the shared tool functions, so the
REST and MCP surfaces never drift.

## MCP server

`pip install "forecast-os-gtm[mcp]"`, register `forecast-os-mcp` with your MCP
client, and an agent gets six tools directly: `preview_panel`, `forecast`,
`compare`, `quota`, `list_models`, and `list_mappings`. See
[connectors.md](connectors.md).

## Foundation models

`pip install "forecast-os-gtm[timegpt]"` registers `timegpt`, a zero-shot
Nixtla-TimeGPT baseline that plugs into every engine feature (compare,
ensembles, the terminal) like any other model — set `NIXTLA_API_KEY` and use
`get_model("timegpt")`. It stays an optional adapter, so the core remains
numpy/pandas/scipy-only.
