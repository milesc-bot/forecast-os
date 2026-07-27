# Changelog

## 0.8.1 (2026-07-27)

Security and robustness fixes for the REST surface. Recommended upgrade for
anyone running `forecast-os-serve`; no API changes for library or MCP users.

- **`POST /preview` no longer accepts `csv_path`.** The field was passed
  straight to `pandas.read_csv`, which resolves URLs as readily as paths, so
  an unauthenticated request could read any file the server process could
  reach, make the server issue outbound requests to an arbitrary host *with
  the fetched body reflected back in the response*, and allocate unbounded
  memory decompressing a remote archive — all before any panel validation ran.
  The field was never documented or tested as part of the REST API; inline
  `records` is unaffected. The MCP surface, which is local and trusted, keeps
  the server-side path affordance.
- **Forecast horizons are bounded** by `core.base.MAX_HORIZON` (10,000). `h`
  sizes the output array before anything else runs, so an unbounded horizon let
  a single ~650-byte request allocate without limit — enough to get the server
  process OOM-killed. The cap sits in `BaseForecaster.predict`, so it protects
  the REST routes, the MCP tools, and direct library callers alike, and it
  fires before allocation. It is far above any real forecasting need.

Both were found by an adversarial audit of the whole repository and are
reproduced by regression tests (`tests/test_serve.py`, `tests/test_contract.py`).

## 0.8.0 (2026-07-27)

Seasonality: multiplicative seasonal ARIMA and multiple-seasonality
decomposition, closing the last modeling gap on the roadmap. The registry
goes from 21 to 24 models. Core dependencies are unchanged (numpy / pandas /
scipy) — no statsmodels.

- **`sarima` / `auto_sarima`**: SARIMA(p,d,q)(P,D,Q)ₘ estimated by conditional
  sum of squares. The seasonal and non-seasonal lag polynomials are convolved
  into equivalent long AR/MA polynomials, so only the `p+q+P+Q` structural
  coefficients are optimized and the multiplicative constraint stays exact;
  the existing CSS estimator and ψ-weight interval theory then apply unchanged.
  Forecast standard errors integrate the ψ sequence through the same
  `(1−B)ᵈ(1−Bᵐ)ᴰ` operator used to difference the data. `auto_sarima` picks `D`
  by seasonal strength, `d` by the variance heuristic on the seasonally
  differenced series, and `(p,q,P,Q)` by AICc. With `m=1` or
  `seasonal_order=(0,0,0)` it reduces bit-exactly to `arima`.
- **`mstl` + `stl_decompose`**: additive multi-seasonal decomposition by
  iterative backfitting — one seasonal component per period plus trend and
  remainder — forecast by adding cyclically-extended seasonals back onto any
  registry base model (default `auto_ets`). Handles weekly *and* yearly
  seasonality in one model. `robust=True` uses cycle-subseries medians.

Fixes from the adversarial verification pass:

- **The CSS fit is now equivariant in the units of `y`.** L-BFGS-B tests
  convergence on the *absolute* projected gradient, but the CSS objective's
  gradient scales as the square of the units — so a series measured in
  millionths stopped at the all-zero starting point and reported
  `converged=True` (observed at λ=1e-4: all coefficients exactly 0, SSE 64%
  worse). The series is now fitted on a unit-scale copy and the intercept
  scaled back, which leaves the optimum untouched (residuals are linear in
  `(w, c)` at fixed coefficients) while giving the optimizer an O(1) problem
  at any magnitude. Applies to `arima`/`auto_arima` as well, which shared the
  defect. Together with the AICc fix below, `auto_sarima` order selection is
  now invariant across λ ∈ [1e-2, 1e2] on every series tested (was 12/12
  flipping).
- **AICc order selection is now scale invariant.** Candidates were scored on
  differing sample sizes (each model's own warm-up), and `n·log(sse/n)` shifts
  by `2n·log(λ)` under a rescaling of `y` — so changing the units of a series
  flipped the selected order (measured: 21% worse holdout MAE, differing in
  40/40 replications). All candidates are now scored on a common window.
- **Seasonal strength is now robust.** Cycle-subseries means and variances have
  a 0% breakdown point, so a single outlier drove `Fs` from 0.963 to 0.600,
  flipping `D` from 1 to 0 on a strongly seasonal series and tripling holdout
  error. Now uses cycle-subseries medians and a MAD-based scale.
- **MSTL honors its base model's data requirement.** The base was fitted
  through `_fit_series`, bypassing the `min_train_size` guard, so a heavier
  base silently returned all-NaN forecasts with no error; a base failing on its
  own terms leaked a raw `AttributeError`. Both now raise `ForecastOSError`.
- **A seasonal period needs two full cycles to be estimated.** At one cycle each
  cycle-subseries cell holds a single point, so the component memorizes noise
  and the intervals collapse — a nominal 95% interval was measured covering 9%.
  Periods without two cycles are dropped with a warning, as over-long ones
  already were.
- **The terminal's season setting reaches the new models.** It was matched only
  against `season_length`, so `sarima`/`auto_sarima` (which take `m`) and
  `mstl` (which takes `periods`) silently stayed on their class defaults.

## 0.7.1 (2026-07-24)

Packaging: first PyPI release. The distribution is published as
**`forecast-os-gtm`** (`pip install forecast-os-gtm`); the import name is
unchanged (`import forecast_os`), as are the `forecast-os` / `-mcp` / `-tui` /
`-serve` console commands. No code changes.

## 0.7.0 (2026-07-24)

Deal-grain analytics and go-public hardening — closing the highest-impact gaps
from the competitive review.

- **Opportunity-grain layer** (`forecast_os.gtm`): `DealScorer` — a calibrated
  logistic win-probability model on deal features — and `weighted_pipeline`,
  a probabilistic pipeline forecast (Σ p·amount with a calibrated interval
  from the Bernoulli variance). The keystone that the aggregate panel contract
  could not express.
- **Pipeline waterfall**: a `deals` snapshot kind plus `pipeline_waterfall`,
  which diffs two point-in-time opportunity snapshots into created / advanced /
  slipped / pushed / won / lost by amount and count.
- **Currency normalization**: `CurrencyNormalizer` / `convert_currency` with a
  guard that refuses to aggregate mixed-currency amounts without a rate table
  (previously a silent correctness landmine).
- **Scenario planning**: a driver-based `Scenario` object (win-rate, ACV, rep
  count …) over the funnel and quota primitives.
- **Funnel anomaly detection**: `detect_anomalies` flags rate/volume breaks
  (rolling z-score) across the segmented panel.
- **Adoption/CI hardening**: `py.typed` (the library now ships its types),
  mocked statsforecast/neuralforecast adapter tests, a core-only install CI
  job, lazier imports, community-health files, and PyPI release automation.

## 0.6.0 (2026-07-24)

Roadmap complete: snapshot history, driver-based ARIMA, foundation-model
adapter, HTTP serving, and a fully interactive terminal.

- **Snapshot store** (`forecast_os.snapshots`, extra `[snapshots]`): capture
  point-in-time panels and forecasts tagged by `as_of` date, then analyze
  week-over-week — how a target period's number moved across snapshots
  (`snapshot_evolution`), and each committed forecast vs the eventual actual
  (`forecast_vs_actual`), the governance audit trail finally persisted.
- **Exogenous ARIMA**: `arima`/`auto_arima` are now driver-aware
  (regression with ARIMA errors) — pass covariate columns and known-future
  `X_df`, same contract as `ridge_lag`.
- **Foundation-model adapter** (extra `[timegpt]`): `TimeGPTAdapter` wraps
  Nixtla's TimeGPT for zero-shot baselines behind the registry, import-guarded
  so the core stays dependency-light.
- **REST serving layer** (`forecast-os-serve`, extra `[serve]`): a FastAPI
  app exposing `/forecast`, `/compare`, `/quota`, `/models`, `/mappings`
  over JSON, reusing the same tool functions as the MCP server.
- **Terminal**: source add/edit and alert-management modals (persisted to the
  workspace) and Enter-to-drill-down from the dashboard into a series
  forecast — the console is now interactive, not read-only.

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
