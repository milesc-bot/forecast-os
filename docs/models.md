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

## Meta-models

- `ensemble` — mean / median / weighted combination; combining diverse models is the
  most reliable accuracy win in forecasting practice.
- `auto_select` — cross-validates candidate models and picks the best **per series**
  (different series often want different models).
- `conformal` — wraps any model with split-conformal calibration for intervals with
  distribution-free coverage.

## Finance

- `garch` / `GARCH11` — GARCH(1,1): tomorrow's variance = ω + α·(today's shock)² +
  β·(today's variance). Captures volatility clustering; forecasts mean-revert to the
  long-run variance ω/(1−α−β).
- `MonteCarloSimulator` — geometric Brownian motion scenario fans, calibrated from
  observed returns (`from_returns`).
- `MarkovRegimeSwitching` — 2-state Gaussian hidden Markov model fitted by EM;
  produces smoothed bull/bear probabilities and regime-conditional expectations.
- `StrategyBacktester` — walk-forward one-step forecasts → long/flat positions →
  Sharpe / Sortino / max drawdown / hit rate, with transaction costs.
