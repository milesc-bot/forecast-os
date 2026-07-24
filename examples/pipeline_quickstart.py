"""Pipeline quickstart: three ways into the engine, all offline.

1. A HubSpot-shaped export through a mapping recipe.
2. A warehouse query through SQLSource (sqlite stands in for DuckDB/Snowflake).
3. The MCP tools an agent would call.

Run:  python examples/pipeline_quickstart.py
"""

import sqlite3

import numpy as np
import pandas as pd

import forecast_os as fos
from forecast_os.connectors import SQLSource, apply_mapping

rng = np.random.default_rng(11)

# --- 1. HubSpot-shaped records -> recipe -> forecast -----------------------
hubspot = pd.DataFrame(
    {
        "closedate": rng.choice(pd.date_range("2023-01-01", "2025-12-31"), 400),
        "amount": rng.lognormal(9.5, 0.5, 400).round(2),
        "dealstage": rng.choice(["closedwon", "closedlost"], 400, p=[0.6, 0.4]),
        "hubspot_owner_id": rng.choice(["101", "102", "103"], 400),
    }
)
panel = apply_mapping(hubspot, "hubspot_deals", id_cols=("owner",), span="panel")
print(f"hubspot_deals recipe: {panel['unique_id'].nunique()} owner series, {len(panel)} rows")

model = fos.get_model("reconciled", model="auto_ets").fit(panel)
fc = model.predict(3, level=[80])
total = fc[fc["unique_id"] == "total"]
print("next-3-month total forecast:")
print(total.to_string(index=False, float_format="{:,.0f}".format))

# --- 2. A warehouse query (sqlite standing in for your warehouse) ----------
con = sqlite3.connect(":memory:")
hubspot.assign(closedate=hubspot["closedate"].astype(str)).to_sql("deals", con, index=False)
wh_panel = SQLSource(
    "SELECT closedate, amount, dealstage FROM deals", con, mapping="hubspot_deals"
).to_panel()
print(f"\nSQLSource: {len(wh_panel)} monthly rows from SQL, "
      f"total ${wh_panel['y'].sum():,.0f} closed-won")

# --- 3. What an MCP agent sees ---------------------------------------------
from forecast_os.mcp.server import list_mappings_tool, preview_panel  # noqa: E402

preview = preview_panel(records=hubspot.assign(
    closedate=hubspot["closedate"].astype(str)).to_dict("records"),
    mapping="hubspot_deals")
print(f"\nMCP preview_panel: {preview['rows']} rows, {preview['series']} series")
print(f"MCP list_mappings: {len(list_mappings_tool())} recipes available")
print('\nRegister with your MCP client:'
      '\n  {"mcpServers": {"forecast-os": {"command": "forecast-os-mcp"}}}')
