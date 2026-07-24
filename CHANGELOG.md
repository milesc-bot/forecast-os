# Changelog

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
