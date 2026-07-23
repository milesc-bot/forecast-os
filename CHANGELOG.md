# Changelog

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
