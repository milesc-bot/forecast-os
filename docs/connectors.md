# Connectors: getting your data in

The data plane has two pieces that compose:

- A **source** fetches raw records (one row per deal, invoice, or event):
  `CSVSource`, `ParquetSource`, `SQLSource`, `RestSource` and its platform
  subclasses (`HubSpotSource`, `PostHogSource`, `StripeSource`,
  `SalesforceSource`).
- A **schema mapping** is a declarative recipe turning a platform's export
  shape into `(unique_id, ds, y)` panel arguments: column renames, row
  filters (e.g. closed-won only), and aggregation defaults. Recipes are
  registered like models — `list_mappings()` shows them, and
  `register_mapping(SchemaMapping(...))` adds your own.

`source.to_panel()` runs both steps; `apply_mapping(df, "hubspot_deals")`
runs just the shaping when you already have a DataFrame.

## Built-in recipes

| Recipe | Fits | Default |
|---|---|---|
| `salesforce_opportunities` | SFDC report/API export (`Close Date`/`Amount`/`Stage`) | closed-won amounts, monthly |
| `hubspot_deals` | HubSpot deal properties (`closedate`/`amount`/`dealstage`) | closedwon amounts, monthly |
| `pipedrive_deals` | Pipedrive deals (`won_time`/`value`/`status`) | won values, monthly |
| `stripe_invoices` / `stripe_charges` | Stripe API objects (amounts in **cents**) | paid/succeeded, monthly |
| `posthog_events` | PostHog events (`timestamp`/`event`) | daily counts per event |
| `ga4_events` | GA4 BigQuery export (`event_date`/`event_name`) | daily counts per event |
| `mixpanel_events` / `amplitude_events` | their raw event exports | daily counts per event |
| `generic_events` | any log with a `date`/`timestamp` column | daily counts, one series |

Every recipe accepts overrides at apply time: `id_cols=("owner",)` splits per
owner (renames for the common owner columns are pre-wired), `freq="W"`
re-buckets, `filters={}` clears the stage filter.

## REST sources

`RestSource` handles token headers, JSON record extraction (`records_path`
dot-paths), and five pagination styles (cursor, offset, page, next-link, and
Stripe's `has_more`/`starting_after`), with a `max_pages` safety cap. The
platform subclasses pin the right endpoints, auth, pagination, and default
mapping — pass a token and call `.to_panel()`. Notes: Salesforce expects an
already-obtained OAuth access token; Stripe amounts are minor units; nothing
is fetched lazily — `fetch()` pulls all pages into memory.

## Warehouses

`SQLSource(query, con)` accepts anything `pandas.read_sql` accepts, so DuckDB,
Snowflake, BigQuery, and Postgres all work with the driver you already have —
no forecast-os dependency added. Point it at your dbt mart, map, forecast.

## MCP server

`forecast-os-mcp` (extra `[mcp]`) exposes the engine to MCP clients with
tools for previewing panels, forecasting, model comparison, and quota
attainment. Inputs are CSV paths or inline records; outputs are plain JSON
records. Register it in your client config:

```json
{"mcpServers": {"forecast-os": {"command": "forecast-os-mcp"}}}
```
