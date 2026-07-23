# Architecture

forecast-os is organized as layered contracts. Each layer only speaks to the one below
through a fixed interface, so any piece can be replaced without touching the rest.

## The data contract

Everything is a **long panel**: a DataFrame with `unique_id` (series key), `ds`
(datetime or integer time index), `y` (float). `forecast_os.validate_panel` normalizes
and enforces it (sorting, dtypes, duplicate and NaN detection). Multi-series is the
default; single series is just a panel with one id.

## The model contract

`BaseForecaster` defines four methods — `fit(df)`, `predict(h, level=None)`,
`fitted_values()`, `get_params()/clone()`. Concrete univariate models subclass
`PerSeriesForecaster` and implement only two hooks:

- `_fit_series(y) -> state dict` — fit one series, optionally include `"fitted"`
  (in-sample one-step predictions) for residual-based intervals
- `_predict_series(state, h) -> ndarray` — point forecasts

The base class handles the per-series loop, future timestamp generation (datetime or
integer), Gaussian intervals from residual std, and an optional `_predict_sigma`
override lets models supply exact variance recursions (ARIMA ψ-weights, Kalman
covariance, ETS formulas).

Meta-models (Ensemble, AutoSelect, Conformal) subclass `BaseForecaster` directly and
accept members as instances **or registry names** — names are resolved at `fit` time.

## The registry

`@register(name, family=...)` puts a class in a process-global registry;
`get_model(name, **params)` constructs it; `list_models()` enumerates. The CLI, engine,
ensembles, and cross-validation all resolve models through the registry, which is what
makes third-party models first-class citizens: register one and every engine feature
works with it, including the shared contract test.

## Evaluation

`cross_validation(df, models, h, n_windows, step_size, level)` implements expanding-
window walk-forward validation: for each cutoff every model is **re-fitted from
scratch** on data up to the cutoff (via `clone()`) and scored on the next `h` points —
no leakage by construction. The output is one frame with a column per model plus
`cutoff`, directly consumable by `evaluate()` for per-series, per-metric scoring.

## Uncertainty

Three tiers, most exact wins:

1. **Model-native variance** — ARIMA, Kalman, SES, naive/drift use their theoretical
   forecast-error variance.
2. **Residual fallback** — models without variance theory use in-sample residual std.
3. **Conformal calibration** — `ConformalForecaster` wraps any model, holds out a
   calibration split per series, and produces distribution-free empirical-quantile
   intervals.

## Finance layer

Financial models don't force the panel contract where it doesn't fit: `GARCH11`,
`MonteCarloSimulator`, and `MarkovRegimeSwitching` work on plain return arrays with
finance-native APIs, while `GARCHVolatility` bridges GARCH into the registry as a
volatility forecaster. `StrategyBacktester` closes the loop: it turns any registered
forecaster into one-step trading signals via walk-forward CV and scores the resulting
strategy with the finance metrics (Sharpe, Sortino, max drawdown, hit rate, ...).
