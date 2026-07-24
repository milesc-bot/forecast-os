# Model guide

How each built-in model works, when to use it, and how its intervals are computed.

## Baselines

| Model | Forecast | Intervals |
|---|---|---|
| `naive` | last value | σ·√k (random-walk growth) |
| `seasonal_naive` | value one season ago | σ·√(seasons elapsed) |
| `drift` | last value + average historical slope | σ·√(k(1+k/(T−1))) |
| `window_average` | mean of trailing window | residual σ |

Always include a baseline in comparisons: a model that can't beat `seasonal_naive` on
seasonal data isn't earning its complexity.

## Exponential smoothing (ETS)

- `ses` — simple exponential smoothing; flat forecast; α optimized by SSE.
- `holt` — adds a linear trend component.
- `holt_winters` — adds additive or multiplicative seasonality
  (`seasonal="add"|"mul"`; multiplicative requires positive data).
- `auto_ets` — fits the applicable candidates and picks by AICc.

Fast, robust, interpretable — the default first real model for business series.
Smoothing parameters are optimized per series, never shared. When optimizing β,
`holt` caps the search at β ≤ 0.3: unguarded SSE optimization on seasonal data
is minimized near β ≈ 1, where the trend collapses to the last first-difference
and the forecast extrapolates a seasonal swing as a runaway linear trend. On
seasonal data, pass `season_length` to `auto_ets` (and to `auto_select`
candidates) so the seasonal candidates can compete.

## Theta

The M3-competition-winning method: decomposes the (deseasonalized) series into a linear
trend line (θ=0) and a curvature-amplified line (θ=2, forecast by SES), then averages.
Strong on trending business data at a fraction of ARIMA's cost.

## ARIMA

`arima` fits ARIMA(p,d,q) by conditional sum of squares with scipy; forecasts invert
the differencing, and intervals use the ψ-weight (MA(∞)) representation, so uncertainty
grows correctly with horizon. `auto_arima` picks `d` by a variance-minimization
heuristic and (p,q) by AICc grid search. Use for autocorrelation-driven series without
strong seasonality (seasonal ARIMA is on the roadmap).

Note: like R's `arima`, `include_mean` applies only when `d == 0` — a differenced
model has no drift constant (its forecast is trend-free by construction);
`auto_arima` can still capture drift-like behavior through AR structure.

## Kalman filter

`kalman` fits a structural state-space model — `local_level` (random walk + noise) or
`local_linear` (stochastic level + trend) — with noise variances estimated by maximum
likelihood via the prediction-error decomposition. Intervals come from exact covariance
propagation. Best for noisy signals needing principled real-time updating.

## RidgeLag (ML)

Ridge regression on lagged values plus optional Fourier seasonal terms, applied
recursively for multi-step forecasts. The "MLForecast pattern" without the dependency —
a strong, fast tabular-ML baseline, and the natural bridge to gradient-boosted
extensions.

## Intermittent demand

- `croston` — the classic method for lumpy series: separate exponential
  smoothing of nonzero demand sizes and inter-demand intervals.
- `tsb` — Teunter-Syntetos-Babai: smooths the demand *probability* every
  period, so obsolescence decays the forecast. Both suit sparse enterprise
  bookings where standard models produce flat lines with negative intervals.

## Meta-models

- `ensemble` — mean / median / weighted combination; combining diverse models is the
  most reliable accuracy win in forecasting practice.
- `auto_select` — cross-validates candidate models and picks the best **per series**
  (different series often want different models). Seasonality-aware: by default
  (`season_length="auto"`) it infers the season from the data frequency
  (daily→7, business-daily→5, monthly→12, quarterly→4, hourly→24) and fields
  seasonal candidates (seasonal naive, seasonal Theta/AutoETS); selection uses
  MASE with train-only scaling (scale-free and safe on zero-crossing series).
- `conformal` — wraps any model with split-conformal calibration for intervals with
  distribution-free coverage.
- `reconciled` — hierarchical reconciliation over path-encoded ids
  (`"west/alice"` rolls up to `"west"` and `"total"`): bottom-up, top-down
  forecast-proportions, or MinT-OLS projection. Guarantees children sum to
  parents at every horizon step; reconciled intervals are a linear-projection
  approximation (documented in the docstring).

## GTM (`forecast_os.gtm`)

- `retention_sbg` — shifted-beta-geometric cohort retention (Fader-Hardie):
  fits churn as a beta-mixture of geometric lifetimes per cohort, with pooled
  parameters shrinking short cohorts; forecasts are monotone and stay in
  [0, 1]. Feed it a cohort triangle via `gtm.cohort_panel`.
- Plus non-registry primitives: `to_panel` (CRM events → panels),
  `stage_panel` / `conversion_rates` / `propagate` (funnel math),
  `attainment_probability` and `pipeline_coverage` (quota questions from
  forecast intervals).

### Deal-grain (one row per opportunity)

- `DealScorer` — a calibrated logistic win-probability model on deal features
  (L2-regularized logistic fit by scipy, Platt-scaled so predicted
  probabilities match empirical win rates). `weighted_pipeline` turns those
  odds into a *probabilistic* pipeline forecast: expected won-$ =
  Σ p·amount with a calibrated interval from the Bernoulli variance
  Σ p(1−p)·amount².
- `pipeline_waterfall` / `waterfall_summary` — diff two point-in-time
  opportunity snapshots (`SnapshotStore` `kind="deals"`) into created /
  advanced / expanded / won / lost / removed by count and signed amount; the
  bridge closes exactly from opening to closing pipeline.
- `Scenario` / `compare_scenarios` — driver-based what-if (top-of-funnel,
  win-rate, ACV, rep count) over the funnel primitives.
- `detect_anomalies` — rolling-z-score flags on rate/volume series across the
  segmented panel (a conversion drop in one region).

## Finance

- `garch` / `GARCH11` — GARCH(1,1): tomorrow's variance = ω + α·(today's shock)² +
  β·(today's variance). Captures volatility clustering; forecasts mean-revert to the
  long-run variance ω/(1−α−β).
- `MonteCarloSimulator` — geometric Brownian motion scenario fans, calibrated from
  observed returns (`from_returns`).
- `MarkovRegimeSwitching` — 2-state Gaussian hidden Markov model fitted by EM;
  produces smoothed bull/bear probabilities and regime-conditional expectations.
- `StrategyBacktester` — walk-forward one-step forecasts → positions → a full
  risk report (Sharpe / Sortino / max drawdown / hit rate / annualized vol /
  Calmar / historic VaR & CVaR), with transaction costs. Three sizing rules:
  `"binary"` (long/flat on the forecast sign), `"proportional"`
  (exposure scaled by `P(r > threshold)` under the forecast distribution —
  uses the model's prediction intervals), and `"kelly"` (fractional Kelly
  `f·ŷ/σ²`, capped at `max_leverage`).
