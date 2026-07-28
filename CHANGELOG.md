# Changelog

## 0.10.1 (2026-07-28)

Fixes a pandas-version incompatibility shipped in 0.10.0. Anyone on pandas 2.x —
which is every supported install below pandas 3 — should upgrade.

- **`detect_anomalies(by=...)` raised `ValueError: Categorical categories cannot
  be null` on pandas 2.x** whenever the grouping column contained a null. The
  0.10.0 fix for silently-unscanned null segments used
  `groupby(..., dropna=False)`, which pandas 2.x rejects for a key column
  holding nulls (with either `sort` setting); pandas 3 accepts it. Grouping now
  runs on factorized codes with `use_na_sentinel=False`, which gives null its
  own bucket on every supported pandas. Null keys remain their own segment and
  still sort last.

The full suite is now run against both pandas 2.2.3 and pandas 3.x before
release, not just the locally installed version — which is what let this reach a
tag. `requires-python >= 3.10` and `pandas >= 2.0` are unchanged.

## 0.10.0 (2026-07-27)

Correctness release, part two. v0.9.0 shipped the must-fix tier of a
whole-repository adversarial audit; this closes the remaining 65 findings.
The theme again is silence — numbers that were wrong without saying so, and
docstrings that promised more than the code delivered.

Every fix carries a regression test naming the defect it prevents. 1836 tests
pass, up from 1561.

### Breaking — a metric was renamed because its name was false

- **`crps` is now `wis`.** The number this library scored under the name
  `crps` was never the Continuous Ranked Probability Score — it is the
  Weighted Interval Score (Bracher et al. 2021), a quadrature rule for the
  CRPS that is only accurate on a dense level set. On the level sets
  forecast-os actually produces it understates the true CRPS badly: the
  measured `wis / crps` ratio is **0.89** at `level=[80]`, **0.61** at
  `[80, 95]`, and **0.76** at `[50, 80, 95]` (1.01 with 49 levels). A number
  labelled `crps` that is 0.6x the CRPS silently corrupts any comparison
  against another library.

  The value is unchanged — only the name and the claims are. `metrics=['crps']`
  still scores everywhere (`evaluate`, `ForecastEngine.compare`, the CLI, the
  MCP tools) and emits a `FutureWarning`, but the returned row is labelled
  `wis`, **so callers that select rows or columns by metric name must ask for
  `wis`**. Renaming was chosen over computing a true CRPS because CRPS is not
  identifiable from K nested intervals plus a median without inventing an
  interpolation assumption. The measured ratios are pinned in the test suite,
  so the understatement is documented rather than hidden.

### Breaking — input that was silently mishandled is now rejected

- **Non-integral confidence levels** raise instead of truncating. `level=[99.9]`
  used to serve an interval ~22% narrower than requested under a `lo-99` label
  nobody asked for. Whole-number levels that arrive with float rounding error
  (`100 * (1 - 0.34)`) are still accepted.
- **`±inf` in `y`** is rejected by `validate_panel`, and `allow_missing` does
  not waive it — it covers NaN only. Infinite targets propagated to
  `sigma = inf` and infinite interval bounds around a finite `yhat`.
- **`seasonality` / `m` must be a positive integer**; `evaluate` raises rather
  than silently producing a meaningless scaled metric.
- **`AutoSelect(metric=...)` rejects signed metrics** (`bias`, `pct_bias`,
  `tracking_signal`) at construction: argmin over a signed metric selects the
  most negatively-biased model, not the best one.
- **`AutoETS` requires 4 observations**, not 3. `min_train_size = 3` was
  unreachable — every 3-row series already failed with "no ETS candidate
  could be fitted".
- **REST/MCP**: `seasonality < 1` and a non-finite `quota` now return HTTP 400
  rather than a 200 carrying a meaningless number.

### Fixed — silently wrong numbers

- **`value_at_risk` / `conditional_var` reported zero risk** when the input
  contained a NaN, while every other metric in the module propagated it.
- **Strategy backtests annualized everything as daily.** `periods=252` was
  hard-coded, so weekly/monthly panels got nonsense Sharpe, Sortino and CAGR.
  Now inferred from `ds` (and honouring `step_size`), with a `periods=`
  override.
- **Monte Carlo drift** applied the Itô correction to a drift its own docs
  described as a log return, biasing every simulated path. The front-page
  README example fed it log returns, exactly as the corrected docstring says
  not to; both are fixed.
- **Conformal intervals under-covered** when the nominal level was
  unattainable from the calibration set — it silently capped at the largest
  residual. It now warns instead of quietly serving a narrower band.
- **RidgeLag reported its one-step residual sigma at every horizon**, so
  multi-step intervals never widened with `h`.
- **`SchemaMapping.apply` string-concatenated** two source columns renamed onto
  one canonical name instead of raising.
- **Zero-padded identifiers** (`"007"`) were destroyed by the numeric-text
  cleaner, silently merging distinct series.

### Fixed — credentials, and requests going where they should not

- **REST connectors no longer send `headers` off-origin.** A `next_url` in a
  paginated response could name any host and receive the `Authorization`
  bearer token. Headers are now withheld across an origin change, with a
  warning naming both origins, and a `next_url` with a non-http(s) scheme is
  rejected rather than handed to `requests`.
- **`POST /forecast` no longer returns a 500 with a traceback** for malformed
  `model_params`. The module docstring and `docs/serving.md` both promised
  "never a 500 traceback"; overrides that survived construction and failed
  inside `fit()` escaped every handler. Present since the REST surface shipped.

### Also

Failed fits no longer leave a model marked fitted on a partial panel
(`PerSeriesForecaster` and `ConformalForecaster`); residual sigma no longer
overflows to `inf` on large finite targets; an object-dtype integer `ds` is no
longer reinterpreted as nanoseconds since the epoch; a hand-edited
`workspace.json` no longer kills the TUI at startup; and the terminal's stale
worker results can no longer overwrite a newer refresh.

## 0.9.0 (2026-07-27)

Correctness release. An adversarial audit of the whole repository found 73
confirmed defects; this ships the must-fix tier. The recurring theme is data
that disappeared or came back wrong **without saying so**, so several calls
that used to succeed quietly now fail loudly — see "Breaking" below.

### Breaking — things that used to succeed and now raise

- **Null keys are rejected by `validate_panel`.** A NaN `unique_id` or `ds`
  passed validation and then vanished: pandas' `groupby` defaults to
  `dropna=True`, so one blank id cell deleted an entire series from the
  forecast with no error, and a null `ds` sorted to the end of its series where
  it was mistaken for the latest observation, silently changing the point
  forecast. `allow_missing` does not waive this — it concerns a missing
  observation, not a row that cannot say what it is.
- **`gtm.events.to_panel` rejects null dates and ids.** It was dropping those
  rows: a four-row frame totalling 100.0 came back totalling 70.0. Null ids
  were worse than dropped — `.astype(str)` turned them into a series literally
  named `"nan"`.
- **The CLI rejects undated rows under `--freq`/`--agg`** instead of discarding
  them (exit code 2, naming the offending row labels). Off-grid timestamps are
  now *bucketed* into their containing period rather than deleted; the
  documented example was losing 96% of its revenue.
- **Non-numeric value columns raise** in `connectors.base` instead of being
  string-concatenated: two deals of $1,000 and $2,000 summed to `10002000.0`.
- **`DealScorer` rejects a nullable target containing `pd.NA`** instead of
  training a degenerate all-zero model that scored every deal at p=0.5.
- **`pipeline_waterfall` rejects null amounts** that would break the
  reconciliation identity (opening + movements == closing).
- **`GARCH11.fit` raises on non-convergence** rather than returning parameters
  that are not the MLE. It gained `converged_` and `opt_message_`.
- **Forecast horizons are capped** at `core.base.MAX_HORIZON` (10,000) — see
  0.8.1.

### Fixed — silently wrong numbers

- **`evaluate()` gave a ~6x wrong MASE/RMSSE on an unsorted panel.** Those
  metrics scale on `mean(|y_t - y_{t-m}|)`, defined only on the chronologically
  ordered series; `cross_validation` was already order-invariant, so the same
  frame produced a correct `cv_df` and a wrong metric. Now order-invariant.
- **Platt calibration made probabilities worse, and `calibrate=True` is the
  default.** On a separable split the unpenalized slope ran to ~1e3 and pinned
  every win probability to 0/1 — out-of-sample log loss 10x worse than
  predicting the base rate. Now uses Platt (1999) smoothed targets, which make
  the objective bounded under separation, fitted on cross-fitted out-of-fold
  scores. Calibration is skipped on samples too small to estimate the map, and
  says so with a warning rather than silently doing nothing.
- **`weighted_pipeline(by=...)` dropped deals with a null segment label**, so
  segments no longer summed to the ungrouped total. The null group is now
  surfaced (with a warning naming the column, the count, and the dollars), so
  segments reconcile.
- **`KalmanForecaster` was not scale-equivariant**: hard-coded log-variance
  bounds meant large- or small-magnitude series got bound-clamped parameters,
  wrong point forecasts and miscalibrated intervals. It now fits a unit-scale
  copy, as `arima`/`sarima` already did.
- **`ShiftedBetaGeometric` anchored cohort age on `y[0]` rather than `ds`**,
  shifting the whole retention curve when age-0 retention was below 1.0.
- **Timezone-aware snapshots were written successfully but were permanently
  unreadable** — `load()` and `history()` both raised `TypeError`.
- **`AutoSelect()` crashed** on any series whose length fell in a narrow band
  just above its CV span.

### Notes

`connectors.base` validates numeric text before stripping separators, so
US-format `"1,000.50"` and accounting negatives `"(500)"` parse while
ambiguous European `"1.000,50"` is rejected rather than silently read as
`1.0005`.

Every fix in this release carries a regression test naming the defect it
prevents. 1561 tests pass.

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
