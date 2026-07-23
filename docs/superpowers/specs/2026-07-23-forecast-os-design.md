# Forecast OS — Design Spec (2026-07-23)

## Goal

An open-source **forecasting engine** ("Forecast OS") that unifies three model families —
classical statistical, machine-learning, and quantitative-financial — behind one data
contract, one model interface, one evaluation harness, and one CLI. Guided by the research
report `I need deep research done into forcasting models a.md` (Perplexity deep-research,
2026), which recommends the Nixtla-style panel data contract, an sktime-style plugin
registry, probabilistic-first outputs, and a built-in walk-forward backtesting engine.

Public repo target: `github.com/milesc-bot/forecast-os`, MIT license, Python ≥ 3.10.

## Key decisions

1. **Self-contained core, optional heavy backends.** The engine implements its models
   natively on numpy/pandas/scipy only. Rationale: an "engine" that is a pile of wrappers
   around statsforecast/neuralforecast/qlib is glue, not an engine; heavy deps (torch)
   make installation brittle and CI slow. The research doc's recommended libraries are
   honored as *optional adapters* (`pip install forecast-os[nixtla]`), import-guarded, so
   the same registry can serve statsforecast/neuralforecast models when installed.
2. **Universal data contract** — long-format pandas DataFrame with columns
   `unique_id` (series key), `ds` (timestamp or integer index), `y` (float value).
   Exactly the Nixtla contract the doc recommends. All models speak only this language.
3. **One interface** — `BaseForecaster.fit(df) -> self`, `predict(h, level=None) ->
   DataFrame(unique_id, ds, yhat[, lo-l, hi-l...])`, plus `fitted_values()` for residual
   analysis. A `PerSeriesForecaster` helper base handles the per-series groupby loop and
   future-date generation so concrete models implement only
   `_fit_series(y) -> state` / `_predict_series(state, h) -> np.ndarray`.
4. **Probabilistic-first** — every model produces prediction intervals: natively where the
   model theory provides them (ETS/ARIMA/Kalman variance recursions, GARCH),
   Gaussian-residual fallback otherwise, and a model-agnostic **conformal** wrapper
   (split-conformal on holdout residuals) for calibrated coverage guarantees.
5. **Registry as the OS kernel** — `@register("name", family=...)` decorator; models are
   discoverable (`list_models()`), constructible by name (`get_model("auto_ets")`), and
   third-party packages can plug in via the same decorator.
6. **Evaluation harness built in** — walk-forward `cross_validation()` (expanding window,
   Nixtla-style output with `cutoff` column) + point metrics (MAE, RMSE, MAPE, sMAPE,
   MASE, RMSSE), probabilistic metrics (pinball loss, interval coverage), and financial
   metrics (Sharpe, Sortino, max drawdown, hit rate, VaR/CVaR, Calmar).
7. **Finance is a first-class layer, not an afterthought** — GARCH(1,1) volatility
   modeling/forecasting, Monte Carlo (GBM) price simulation, 2-state Gaussian Markov
   regime switching (Hamilton filter + EM), and a forecast-driven strategy backtester
   that turns any registered forecaster into a long/flat trading signal and scores it
   with the financial metrics.
8. **Interfaces: SDK + CLI.** `ForecastEngine` facade (`forecast`, `cross_validate`,
   `compare` leaderboard) and an argparse CLI (`forecast-os forecast|compare|models|simulate`)
   reading/writing CSV. REST serving is out of scope for v0.1 (documented as roadmap) —
   YAGNI; the CLI + SDK cover the open-source use case.
9. **No neural nets in core v0.1.** Torch-based TFT/N-HiTS/PatchTST are exactly what the
   optional neuralforecast adapter provides; reimplementing them poorly would be worse
   than adapting the best-in-class implementations. The core's ML family is a
   lag-feature + calendar-feature ridge autoregressor (the "MLForecast pattern" without
   the dependency), which benchmarks respectably and keeps core install < 100 MB.

## Architecture (mirrors the research doc's layer diagram)

```
┌───────────────────────────────────────────────┐
│ Interface: ForecastEngine SDK · argparse CLI  │
├───────────────────────────────────────────────┤
│ Model Registry (@register · get_model · list) │
├──────────────┬───────────────┬────────────────┤
│ Statistical  │ ML            │ Quant/Finance  │
│ Naive/SNaive │ RidgeLag      │ GARCH(1,1)     │
│ Drift/WinAvg │ Ensemble      │ Monte Carlo    │
│ SES/Holt/HW  │ AutoSelect    │ Regime-Switch  │
│ AutoETS      │               │ Strategy BT    │
│ Theta        │               │                │
│ ARIMA/AutoARIMA │            │                │
│ Kalman (local level/trend)   │                │
├──────────────┴───────────────┴────────────────┤
│ Uncertainty: native variance · conformal      │
├───────────────────────────────────────────────┤
│ Preprocessing: impute · scale · log · diff ·  │
│ calendar & Fourier features · Pipeline        │
├───────────────────────────────────────────────┤
│ Evaluation: walk-forward CV · point/prob/fin  │
│ metrics · leaderboard                         │
├───────────────────────────────────────────────┤
│ Data: contract validation · synthetic gens ·  │
│ embedded AirPassengers · CSV I/O              │
└───────────────────────────────────────────────┘
```

## Package layout

```
src/forecast_os/
  __init__.py            public API re-exports, __version__ = "0.1.0"
  core/types.py          validate_panel(), infer_freq, future_ds(), panel helpers
  core/base.py           BaseForecaster ABC, PerSeriesForecaster
  core/registry.py       register/get_model/list_models
  core/exceptions.py     ForecastOSError, DataContractError, NotFittedError
  preprocessing/transforms.py  Imputer, StandardScalerT, LogTransform, Differencer
  preprocessing/calendar.py    calendar_features(), fourier_features()
  preprocessing/pipeline.py    Pipeline (fit/transform/inverse_transform)
  models/baselines.py    Naive, SeasonalNaive, Drift, WindowAverage
  models/ets.py          SES, Holt, HoltWinters (add/mul seasonality), AutoETS
  models/theta.py        Theta (classic two-theta-line decomposition)
  models/arima.py        ARIMA(p,d,q) via CSS + scipy optimize, AutoARIMA (AICc grid)
  models/kalman.py       KalmanFilter forecaster (local level / local linear trend)
  models/ml.py           RidgeLagForecaster (lags + calendar features, closed-form ridge)
  models/ensemble.py     Ensemble (mean/median/weighted)
  models/auto.py         AutoSelect (CV-based model selection)
  finance/garch.py       GARCH(1,1) MLE + GARCHVolatility forecaster
  finance/montecarlo.py  MonteCarloSimulator (GBM, from_returns calibration)
  finance/regime.py      MarkovRegimeSwitching (2-state Gaussian, EM + Hamilton filter)
  finance/metrics.py     sharpe_ratio, sortino_ratio, max_drawdown, hit_rate, var, cvar, calmar
  finance/backtest.py    StrategyBacktester (forecast-driven long/flat)
  uncertainty/conformal.py  ConformalForecaster wrapper
  evaluation/metrics.py  mae, rmse, mape, smape, mase, rmsse, pinball_loss, coverage, evaluate()
  evaluation/backtest.py cross_validation() walk-forward
  datasets/synthetic.py  generate_series(), generate_returns()
  datasets/air_passengers.py  load_air_passengers()
  adapters/statsforecast_adapter.py, neuralforecast_adapter.py  (import-guarded)
  engine.py              ForecastEngine facade
  cli.py                 argparse CLI, entry point `forecast-os`
tests/                   pytest suite mirroring modules (~22 files)
```

## Testing strategy

- Unit tests per module; statistical models tested by **parameter recovery on synthetic
  data** (e.g., AR(1) φ recovered within tolerance; GARCH persistence recovered on
  simulated GARCH data; regime model separates two planted regimes) and by
  **beating the naive baseline** on structured synthetic series (trend/seasonal).
- Interval tests assert approximate coverage on Gaussian synthetic data (wide tolerance).
- Contract tests: every registered model round-trips the panel contract on a 3-series
  panel, returns exactly h rows per series, correct columns, monotone future `ds`.
- Deterministic: all randomness through seeded `np.random.default_rng`.
- CI: GitHub Actions, ruff + pytest on Python 3.10–3.13 (ubuntu) — 3.14 tested locally.

## Out of scope for v0.1 (documented roadmap)

REST API service, native deep-learning models, TimeGPT/foundation-model adapter,
hierarchical reconciliation, exogenous regressors in ARIMA, dashboard UI.
