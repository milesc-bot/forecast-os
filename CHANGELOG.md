# Changelog

## 0.5.0 (2026-07-24)

The terminal: an always-on console scaffold.

- **TUI** (`forecast-os-tui`, extra `[terminal]`): a keyboard-driven textual
  console over the engine — dashboard watchlist with 1-step forecasts and
  sparklines, forecast fan-chart screen, model leaderboard, governance
  screen (signed bias + coverage, sandbagging highlighted), sources view;
  persistent workspace at `~/.forecast-os/workspace.json`; auto-refresh via
  a background worker; alert rules (`forecast_below`, `coverage_below`);
  `--demo` mode that boots on seeded GTM data instantly.

## 0.4.0 (2026-07-24)

The data plane: plug your pipeline in.

- **Connectors** (`forecast_os.connectors`, extra `[connectors]`): a `Source`
  + `SchemaMapping` contract with registered recipes for Salesforce,
  HubSpot, Pipedrive, Stripe, PostHog, GA4, Mixpanel, and Amplitude export
  shapes; `CSVSource`/`ParquetSource` with conservative currency cleaning;
  `RestSource` with cursor/offset/page/next-link pagination plus thin
  `HubSpotSource`, `PostHogSource`, `StripeSource`, `SalesforceSource`
  clients; `SQLSource` for any warehouse pandas can read (bring your own
  driver: DuckDB, Snowflake, BigQuery, Postgres).
- **MCP server** (`forecast-os-mcp`, extra `[mcp]`): exposes the engine to
  any MCP client (Claude Desktop/Code, agents) — preview panels, forecast,
  compare, and quota-attainment tools over CSV paths or inline records.
- **CLI**: `--mapping hubspot_deals` applies a platform recipe directly;
  new `mappings` subcommand lists available recipes.

## 0.3.0 (2026-07-24)

The GTM wave: driver-based forecasting, coherent hierarchies, and a
go-to-market domain layer.

- **Exogenous covariates**: extra numeric panel columns now reach models that
  declare `supports_exog` (RidgeLag first); `predict(h, X_df=...)` takes known
  future drivers; cross-validation threads held-out covariates automatically;
  models that ignore covariates now warn instead of staying silent.
- **Model persistence**: `model.save(path)` / `forecast_os.load(path)` with a
  version-stamped envelope; every registered model contract-tested to
  round-trip.
- **Hierarchical reconciliation**: `reconciled` meta-model (bottom-up,
  top-down forecast-proportions, MinT-OLS) over path-encoded ids
  (`"west/alice"`), plus `aggregate_panel`; rep, team, and total forecasts
  now add up.
- **Governance**: signed `bias` / `pct_bias` / `tracking_signal` metrics;
  `evaluate(by="cutoff")` rolling breakdowns; `compare()` degrades per model
  instead of failing atomically; parameterized model specs
  `("ridge_lag", {"lags": 12})`.
- **GTM layer** (`forecast_os.gtm`): `to_panel` event→panel aggregation (the
  CRM-export bridge), funnel stage/conversion/propagation, quota attainment
  probability from forecast intervals, cohort retention with a pooled
  shifted-beta-geometric model (`retention_sbg`).
- **New models**: Croston and TSB for intermittent/lumpy demand.
- **Fiscal calendars**: `FiscalCalendar` (FY start month, 4-4-5) with
  quarter features; `LogitTransform` for bounded rates; `fill_gaps` for
  irregular exports.
- **CLI**: `--id-col/--time-col/--target-col/--agg/--freq` mapping flags so
  raw CRM exports load directly; `--param key=value` model parameters.

## 0.2.0 (2026-07-23)

Probabilistic evaluation, honest AutoML defaults, and a real risk loop.

- **Interval metrics in the standard pipeline**: `evaluate()` and
  `ForecastEngine.compare(level=[...])` now score prediction intervals —
  `coverage`, `winkler`, `pinball` (per level) and a quantile-approximated
  `crps` — consuming the `{model}-lo/hi-{level}` columns cross-validation
  already emits. New `winkler_score` function. CLI `compare` gains `--level`.
- **AutoSelect is seasonality-aware**: new `season_length` parameter
  (default `"auto"` infers from the data frequency: daily→7, monthly→12, …)
  adds seasonal candidates (seasonal naive, seasonal Theta/AutoETS); the
  default selection metric is now MASE with train-only scaling instead of
  sMAPE (which saturates on zero-crossing series).
- **Probabilistic strategy backtesting**: `StrategyBacktester` gains
  `sizing="binary"|"proportional"|"kelly"` — proportional and Kelly sizing
  consume the model's prediction intervals — plus a risk report in the
  summary (annualized vol, Calmar, historic VaR/CVaR at 95%).

## 0.1.0 (2026-07-23)

Initial release.

- Universal `(unique_id, ds, y)` panel data contract with validation
- Plugin model registry (`register` / `get_model` / `list_models`)
- Baselines: naive, seasonal naive, drift, window average
- Statistical: SES, Holt, Holt-Winters (add/mul), AutoETS, Theta,
  ARIMA (CSS) + AutoARIMA, Kalman filter (local level / local linear trend)
- ML: ridge autoregression with lag + Fourier features
- Meta: ensemble (mean/median/weighted), CV-based AutoSelect, split-conformal
  prediction intervals
- Finance: GARCH(1,1) volatility modeling + forecasting, GBM Monte Carlo
  simulation, 2-state Markov regime switching, risk/performance metrics
  (Sharpe, Sortino, max drawdown, VaR/CVaR, Calmar, hit rate), forecast-driven
  strategy backtester
- Preprocessing: imputation, scaling, log transform, differencing, calendar &
  Fourier features, pipelines with forecast-aware inverse transforms
- Evaluation: walk-forward cross-validation, MAE/RMSE/MAPE/sMAPE/MASE/RMSSE,
  pinball loss, interval coverage
- `ForecastEngine` facade with model comparison leaderboard
- `forecast-os` CLI: forecast, compare, models, simulate
- Optional statsforecast / neuralforecast adapters
