# Forecast OS Implementation Plan

> **For agentic workers:** Executed via parallel workflow subagents (subagent-driven
> development adapted for file-disjoint parallel tasks). Each task is implemented by one
> agent that reads: this plan section, the design spec
> (`docs/superpowers/specs/2026-07-23-forecast-os-design.md`), and the committed core
> modules. TDD per task: write tests first, run them (fail), implement, run until green.
> **Agents do not commit and do not edit shared files** (`__init__.py` of shared
> packages, core/, evaluation/, pyproject) — integration is centralized after the fan-out
> to avoid git lock contention and import races.

**Goal:** Implement all model families, the finance layer, uncertainty wrapper,
preprocessing, engine facade, and CLI of forecast-os against the committed core contracts.

**Architecture:** Every model subclasses `PerSeriesForecaster` (or `BaseForecaster` for
meta-models) from `src/forecast_os/core/base.py`, registers via
`@register(name, family=...)`, and is exercised by the registry-driven contract test
`tests/test_contract.py`. Numpy/pandas/scipy only.

**Tech stack:** Python ≥3.10, numpy, pandas, scipy; pytest; ruff (line-length 100,
select E,F,W,I,B,UP).

## Global constraints

- Data contract: long panel `(unique_id, ds, y)`; validate with
  `forecast_os.core.types.validate_panel`. Never assume a single series.
- Constructor args MUST be stored as same-named attributes (required by
  `BaseForecaster.get_params()/clone()`).
- All randomness through `np.random.default_rng(seed)` with a `seed` constructor param.
- Every registered model must work with **default constructor params** on the contract
  panel (3 daily series, length 80, weekly seasonality) — including `predict(h, level=[80])`
  producing finite values, and raising `NotFittedError` (a `ForecastOSError`) on
  predict-before-fit.
- Interval columns: `lo-{level}`, `hi-{level}`. Point column: `yhat`.
- Test with `.venv/bin/pytest tests/<your files> -q`; lint with
  `.venv/bin/ruff check <your files>` — both must pass before reporting done.
- No new dependencies. No edits outside your task's file list.

## Registry names (canonical)

| name | class | family | module |
|---|---|---|---|
| naive | Naive | baseline | models/baselines.py |
| seasonal_naive | SeasonalNaive | baseline | models/baselines.py |
| drift | Drift | baseline | models/baselines.py |
| window_average | WindowAverage | baseline | models/baselines.py |
| ses | SES | statistical | models/ets.py |
| holt | Holt | statistical | models/ets.py |
| holt_winters | HoltWinters | statistical | models/ets.py |
| auto_ets | AutoETS | statistical | models/ets.py |
| theta | Theta | statistical | models/theta.py |
| arima | ARIMA | statistical | models/arima.py |
| auto_arima | AutoARIMA | statistical | models/arima.py |
| kalman | KalmanForecaster | statistical | models/kalman.py |
| ridge_lag | RidgeLag | ml | models/ml.py |
| ensemble | Ensemble | ensemble | models/ensemble.py |
| auto_select | AutoSelect | ensemble | models/auto.py |
| conformal | ConformalForecaster | ensemble | uncertainty/conformal.py |
| garch | GARCHVolatility | financial | finance/garch.py |

---

### Task A: Baselines + ETS + Theta

**Files:** Create `src/forecast_os/models/baselines.py`, `src/forecast_os/models/ets.py`,
`src/forecast_os/models/theta.py`, `tests/test_baselines.py`, `tests/test_ets.py`,
`tests/test_theta.py`.

**Consumes:** `PerSeriesForecaster` hooks `_fit_series(y)->dict` /
`_predict_series(state,h)->np.ndarray` / optional `_predict_sigma(state,h)`; state dict may
include `"fitted"` (len == len(y), NaN warm-up). `state["_sigma"]` (residual std) is
available inside `_predict_sigma`.

**Models:**
- `Naive()` — repeat last value. fitted = y shifted by 1.
  `_predict_sigma = sigma * sqrt(arange(1,h+1))` (random-walk variance growth).
- `SeasonalNaive(season_length=7)` — `yhat[i] = y[-m + (i % m)]`; fitted = y shifted by m;
  sigma `= sigma * sqrt(floor(arange(h)/m) + 1)`. Series shorter than m: raise
  `ForecastOSError` (set `min_train_size` dynamically in `__init__` is fine).
- `Drift()` — `y_T + slope*(1..h)`, `slope=(y_T-y_1)/(T-1)`; fitted one-step:
  `y[t-1] + (y[t-1]-y[0])/(t-1)` for t≥2; sigma
  `= sigma*sqrt(k*(1+k/(T-1)))`, k=1..h. min_train_size=2.
- `WindowAverage(window=7)` — mean of last `window`; fitted = trailing-window mean
  (expanding for t<window, NaN at t=0).
- `SES(alpha=None)` — `l_t = a*y_t+(1-a)*l_{t-1}`, `l_0=y[0]`, fitted[t]=l_{t-1};
  if alpha None optimize SSE over [0.01,0.99] via `scipy.optimize.minimize_scalar(bounded)`.
  sigma `= sigma*sqrt(1+(k-1)*a^2)`, k=1..h. Store chosen alpha in state (e.g.
  `state["alpha_"]`), NOT on self (fit must stay per-series).
- `Holt(alpha=None, beta=None)` — level+trend recursions
  (`l_t=a*y_t+(1-a)(l_{t-1}+b_{t-1})`, `b_t=B(l_t-l_{t-1})+(1-B)b_{t-1}`), init
  `l_0=y[0], b_0=y[1]-y[0]`; joint SSE optimization (`scipy.optimize.minimize`,
  L-BFGS-B, bounds [0.01,0.99]^2) when params None. Forecast `l_T + k*b_T`. min_train_size=3.
- `HoltWinters(season_length=7, seasonal="add", alpha=None, beta=None, gamma=None)` —
  standard additive/multiplicative recursions; init from first two seasons (level=mean of
  season 1, trend=(mean season2−mean season1)/m, seasonal indices from season 1 relative
  to its mean); optimize SSE over the None params. `seasonal="mul"` requires all y>0 else
  `ForecastOSError`. min_train_size = 2*m.
- `AutoETS(season_length=None)` — fit candidates: SES, Holt, plus HoltWinters add (and
  mul if y>0) when `season_length` and 2*m ≤ len(y); pick min
  `AICc = n*log(SSE/n) + 2k + 2k(k+1)/(n-k-1)` (k = #free smoothing params + #init states).
  Delegate predict/sigma to the winner's state (store winner per series in state).
- `Theta(season_length=None, theta=2.0)` — classical Theta: if season_length>1,
  multiplicative deseasonalization by seasonal indices (mean of y per seasonal position /
  overall mean; guard y>0 else fall back to additive); theta-0 line = OLS linear trend
  extrapolation; theta line `Q = theta*y_deseas + (1-theta)*line0`; SES-forecast Q; final
  forecast = `(1/theta)*ses_forecast + (1-1/theta)*line0_extrapolation`; reseasonalize.

**Tests (write first, must fail before implementation):**
- Exact-arithmetic cases: Naive/SeasonalNaive/Drift/WindowAverage forecasts on tiny
  hand-computed series (e.g. y=[1,2,3,4,5,6]: naive→[6]*h; drift h=2→[7,8];
  seasonal_naive m=3 h=4 → [4,5,6,4]).
- SES with alpha=1 equals naive; optimized alpha on near-constant series produces
  near-constant forecast; SES flat forecast property (all h equal).
- Holt recovers a clean linear trend (forecast within rtol 5% of true continuation).
- HoltWinters (add & mul) on `generate_series(seasonality=7)` beats Naive MAE on an
  80/20 train/test split by ≥20%; mul raises on negative data.
- AutoETS picks a seasonal candidate on strongly seasonal data (winner class name check)
  and runs fine with season_length=None.
- Theta beats Naive on trending data; with seasonality beats Naive on seasonal panel.
- Interval sanity: SES `predict(h=10, level=[80])` widths non-decreasing in h.

### Task B: ARIMA + Kalman

**Files:** Create `src/forecast_os/models/arima.py`, `src/forecast_os/models/kalman.py`,
`tests/test_arima.py`, `tests/test_kalman.py`.

**Consumes:** same PerSeriesForecaster hooks as Task A.

- `ARIMA(order=(1,1,1), include_mean=True)` — difference d times → w. CSS estimation:
  recursion `e_t = w_t - c - Σφ_i w_{t-i} - Σθ_j e_{t-j}` (e pre-sample = 0, first
  max(p,q) terms conditioned), minimize SSE over (c, φ, θ) with
  `scipy.optimize.minimize(L-BFGS-B)`, φ/θ bounds (-0.99, 0.99), c only when d==0 and
  include_mean. Forecast: iterate recursion with future e=0, then invert differencing
  (cumsum with stored last d values). fitted: predicted w re-integrated (NaN warm-up).
  sigma: psi-weights `ψ_0=1, ψ_j = θ_j·[j≤q] + Σ_{i≤min(j,p)} φ_i ψ_{j-i}`; for the
  d-times-integrated process cumulatively sum psi d times;
  `sigma_k = resid_std * sqrt(Σ_{i<k} Ψ_i²)`. min_train_size = max(p,q)+d+5.
- `AutoARIMA(max_p=3, max_d=2, max_q=3)` — choose d = argmin over 0..max_d of
  `std(diff^d(y)) * 1.05^d` (penalize over-differencing); grid (p,q) with AICc from CSS
  SSE (`k = p+q+1`), skip non-converged fits; delegate to best ARIMA state.
- `KalmanForecaster(model="local_level")` — models: "local_level" (state=[level],
  F=[[1]], H=[1]) and "local_linear" (state=[level,trend], F=[[1,1],[0,1]], H=[1,0]).
  Noise variances (obs R, state Q diagonal) estimated by MLE on the prediction-error
  decomposition (`scipy.optimize.minimize` over log-variances, init log(var(diff(y)))).
  Init x0: [y0] or [y0, mean(diff(y[:10]))], P0 = 1e4*var(y)*I. Forecast: propagate state
  h steps; **native `_predict_sigma` from propagated `H P_k H' + R`** (growing intervals).
  fitted = one-step predictions from the filter pass.

**Tests:** AR(1) simulation (φ=0.7, n=500, seed): fitted ARIMA((1,0,0)) recovers φ within
±0.1. ARIMA(0,1,0)+mean≈drift-like behavior sanity. AutoARIMA on AR(1) picks p≥1,d=0;
on random walk picks d≥1. Forecast of ARIMA(1,0,0) converges toward series mean.
Kalman local_level on noisy constant ≈ constant; local_linear tracks linear trend
(forecast within 5% of continuation); Kalman interval width grows with h. Both models
beat Naive MAE on trending synthetic data.

### Task C: Finance layer

**Files:** Create `src/forecast_os/finance/garch.py`, `finance/montecarlo.py`,
`finance/regime.py`, `finance/metrics.py`, `finance/backtest.py`, and
`tests/test_garch.py`, `tests/test_montecarlo.py`, `tests/test_regime.py`,
`tests/test_financial_metrics.py`, `tests/test_strategy_backtest.py`.
(`finance/__init__.py` is wired at integration — do not edit it.)

**Produces (exact signatures):**
- `garch.GARCH11` — `.fit(returns: np.ndarray) -> self` (demeans internally; MLE of
  (omega, alpha, beta) minimizing `0.5*Σ(log σ²_t + r²_t/σ²_t)`, σ²_0=var(r), L-BFGS-B
  with bounds and alpha+beta≤0.999 penalty); attrs `omega_, alpha_, beta_, mu_,
  cond_vol_` (np.ndarray of in-sample conditional vol);
  `.forecast_variance(h) -> np.ndarray` via
  `σ²_{T+k} = σ²_LR + (α+β)^{k-1} (σ²_{T+1} − σ²_LR)`, `σ²_LR = ω/(1−α−β)`;
  `.forecast_volatility(h)` = sqrt.
- `garch.GARCHVolatility(PerSeriesForecaster)` registered `"garch"`, family financial —
  **standardizes each series (z-score) before MLE and rescales vol forecasts back** so it
  is safe on arbitrary scales (the contract panel is levels ~100). `yhat` = per-period
  conditional volatility forecast; fitted = in-sample conditional vol (aligned, NaN t=0).
- `montecarlo.MonteCarloSimulator(mu=0.0, sigma=0.01, seed=0)` with
  `from_returns(returns) -> MonteCarloSimulator` (classmethod; estimates per-period mu,
  sigma), `.simulate(s0: float, h: int, n_paths: int = 1000) -> np.ndarray` shape
  (n_paths, h), GBM step `S*exp((mu−sigma²/2)+sigma*Z)`, and
  `.summary(s0, h, n_paths=1000, levels=(5,25,50,75,95)) -> pd.DataFrame` with columns
  `step, q05, q25, q50, q75, q95` (quantiles across paths; column per level, `q{level:02d}`).
- `regime.MarkovRegimeSwitching(n_states=2, max_iter=200, tol=1e-6, seed=0)` —
  `.fit(returns) -> self` Gaussian HMM via EM with scaled forward-backward; attrs
  `means_, stds_, transition_ (K×K), smoothed_probs_ (T×K ndarray),
  regimes_` (Viterbi-free: argmax smoothed); `.predict_proba(h) -> np.ndarray` (K,)
  state probabilities h steps ahead (`p_T @ P^h`); `.expected_return(h) -> float`;
  states ordered by mean ascending (state 0 = "bear") — reorder after EM.
- `metrics`: `annualized_return(returns, periods=252)`, `annualized_vol`,
  `sharpe_ratio(returns, rf=0.0, periods=252)`, `sortino_ratio`,
  `max_drawdown(returns) -> float ≤ 0` (on cumprod equity),
  `hit_rate(returns)`, `directional_accuracy(y, yhat)` (sign agreement),
  `value_at_risk(returns, level=0.95) -> float ≥ 0` (historic),
  `conditional_var(returns, level=0.95)`, `calmar_ratio(returns, periods=252)`.
  All accept array-likes; raise ValueError on empty. sharpe/sortino return nan when the
  denominator is ~0.
- `backtest.StrategyBacktester(model, threshold=0.0, cost_bps=0.0)` — `.run(df,
  test_size=60, step_size=1) -> BacktestResult`; walk-forward one-step forecasts via
  `forecast_os.evaluation.backtest.cross_validation(df, [model], h=1,
  n_windows=test_size, step_size=step_size)`; long/flat: pos=1 iff yhat>threshold;
  `strat_ret = pos*y − cost_bps/1e4 * |Δpos|`. `BacktestResult` dataclass: `.summary`
  (dict: total_return, annualized_return, sharpe, sortino, max_drawdown, hit_rate,
  n_trades, exposure), `.frame` (per-period DataFrame: unique_id, ds, y, yhat, position,
  strategy_return, equity). Multi-series: metrics per unique_id (summary dict keyed by
  uid) — keep it simple: `.summary` is a DataFrame with one row per unique_id.

**Tests:** GARCH11 on `generate_returns(garch=(2e-6,0.08,0.85), length=1200)`: recovered
`alpha_+beta_` in [0.8, 0.99], long-run vol within 50% of true
`sqrt(omega/(1-a-b))`; forecast_variance monotone toward long-run. MC: with sigma→1e-9
paths ≈ deterministic drift; quantile monotonicity q05<q50<q95; reproducible with seed;
summary shape. Regime on planted data (600 obs bull μ=+0.002/σ=0.005 then 600 bear
μ=−0.002/σ=0.02): recovered means ordered, smoothed-prob segment accuracy > 0.8,
rows of `transition_` sum to 1. Metrics: hand-computed cases (constant positive returns
→ maxdd 0, sharpe>0; alternating ±1% → hit_rate 0.5); VaR/CVaR on known small array.
Backtester: on returns with strong positive drift, long/flat with threshold 0 ≈ buy&hold
(total_return within tolerance, exposure≈1); with cost_bps>0 total return decreases;
equity column = cumprod(1+strategy_return).

### Task D: ML, Ensemble, AutoSelect, Conformal

**Files:** Create `src/forecast_os/models/ml.py`, `models/ensemble.py`, `models/auto.py`,
`src/forecast_os/uncertainty/conformal.py`, and `tests/test_ml.py`, `tests/test_ensemble.py`,
`tests/test_auto.py`, `tests/test_conformal.py`.

**Interface rules for meta-models (Ensemble/AutoSelect/Conformal):** subclass
`BaseForecaster` directly; accept member models as **instances or registry-name strings**;
resolve strings via `get_model(name)` **inside fit()** (never at import/construct time —
avoids import-order races); `clone()` must deep-clone member instances (strings pass
through). Do not import sibling model modules — your tests define local dummy
forecasters (tiny `PerSeriesForecaster` subclasses) instead.

- `RidgeLag(lags=14, alpha=1.0, season_length=None, fourier_k=3)` (PerSeriesForecaster) —
  features per series: lag matrix y_{t-1..t-lags}, plus Fourier terms
  `sin/cos(2πk t/season_length)` for k=1..fourier_k when season_length; standardize
  features and target (store means/stds in state); closed-form ridge
  `w=(X'X+αI)^{-1}X'y` via `np.linalg.solve` (don't penalize the intercept — center
  instead). Multi-step: recursive (append prediction, rebuild lag row, keep extending t
  for Fourier terms). min_train_size = lags + 10.
- `Ensemble(models=("naive", "ses", "drift"), mode="mean", weights=None)` — fit clones of
  members on df; predict each with the same levels; combine `yhat` by mean/median/weights
  and interval bounds the same way. `name` = "Ensemble". Validate mode; weights must
  match len(models) and sum>0 (normalize).
- `AutoSelect(candidates=("naive","drift","ses","holt","theta","auto_ets","window_average"),
  metric="smape", val_h=12, n_windows=2)` — at fit: run
  `evaluation.backtest.cross_validation(df, resolved_candidates, h=val_h,
  n_windows=n_windows)`, score with `evaluation.metrics.evaluate`, pick the best model
  **per series** (lowest mean metric across windows), refit each winning model class on
  the full panel, store per-series winner; predict stitches winners' rows per series;
  expose `.best_models_` dict uid→model name. If the panel is too short for the CV span,
  fall back to scoring on a single 75/25 holdout (still per series).
- `ConformalForecaster(model="ses", level_calibration_fraction=0.25, min_calibration=8)`
  registered `"conformal"` — split-conformal: per series, hold out the last
  max(min_calibration, frac·T) points; fit member clone on the head, forecast the
  holdout length, collect absolute residuals per series; refit member on full series.
  `predict(h, level)`: point = member forecast; bounds = yhat ± empirical
  `level`-quantile of that series' calibration |residuals| (per-series quantiles;
  quantile via `np.quantile(abs_resid, level/100)`).

**Tests:** RidgeLag: on a pure AR(1) (φ=0.8) beats Naive RMSE on holdout; on seasonal
panel with season_length=7 beats Naive by ≥20% MAE; recursive forecast finite for h=30.
Ensemble: mean of two constant dummies = average; median with 3 dummies; weights
respected; string members resolved at fit (register a dummy in the test, refer by name).
AutoSelect: given dummies where one is exact and one is terrible, `.best_models_` picks
the exact one per series; works on `panel` fixture end-to-end. Conformal: on Gaussian
noise series, empirical coverage of 80% intervals on a test split within [0.6, 0.95];
bounds ordering lo≤yhat≤hi; per-series widths differ when series noise differs (2-series
panel with σ=0.5 vs σ=5).

### Task E: Preprocessing

**Files:** Create `src/forecast_os/preprocessing/transforms.py`,
`preprocessing/calendar.py`, `preprocessing/pipeline.py`, `tests/test_preprocessing.py`.

- Transforms are panel→panel, per-series statistics, sklearn-style:
  `fit(df) -> self`, `transform(df) -> df`, `fit_transform`, `inverse_transform(df)`.
  `inverse_transform` must handle forecast frames too: apply the inverse to every column
  in `{y, yhat}` ∪ columns matching `lo-*`/`hi-*`/`*-lo-*`/`*-hi-*` that are present.
  - `Imputer(method="interpolate")` — methods: interpolate (linear, then
    ffill/bfill edges), ffill, mean. Works with `validate_panel(df, allow_missing=True)`.
    No inverse (inverse_transform = identity).
  - `StandardScaler()` — per-series (y−μ)/σ (σ guard 1e-12). Inverse restores scale; for
    unseen unique_id in inverse: raise ForecastOSError.
  - `LogTransform(offset="auto")` — log(y+offset); "auto" picks 0 when min>0 else
    1−min(y) per series; inverse exp−offset.
  - `Differencer(d=1)` — per-series differencing dropping the first d rows; inverse
    integrates using stored last values (only valid for frames that continue the
    training series — document this).
- `calendar.calendar_features(df, features=("dayofweek","month","dayofyear"),
  cyclical=True) -> df` — appends numeric columns; cyclical adds sin/cos pairs
  (`{f}_sin`, `{f}_cos`) instead of raw ints. Requires datetime ds (raise otherwise).
  `calendar.fourier_features(df, season_length, k=3) -> df` — appends
  `fourier_s{season_length}_{sin|cos}{i}` computed on the per-series integer position.
- `pipeline.Pipeline(steps: list[tuple[str, transform]])` — fit/transform in order,
  inverse_transform in reverse; `named_steps` dict; steps validated unique names.

**Tests:** round-trips (`inverse(transform(df)) ≈ df`) for scaler/log/differencer;
imputer fills all NaN under each method; forecast-frame inverse (build a fake forecast
frame with yhat/lo-80/hi-80, scale-inverse restores magnitudes); calendar column
presence/values (a known Monday → dayofweek 0); fourier periodicity; pipeline order
(log→scale inverse restores original); pipeline rejects duplicate step names.

### Task F: Engine facade + CLI + adapters

**Files:** Create `src/forecast_os/engine.py`, `src/forecast_os/cli.py`,
`src/forecast_os/adapters/statsforecast_adapter.py`,
`src/forecast_os/adapters/neuralforecast_adapter.py`, `tests/test_engine.py`,
`tests/test_cli.py`. (adapters/__init__.py wired at integration; leave it.)

- `ForecastEngine(models=("auto_ets",), level=None)`:
  - `.forecast(df, h, models=None, level=None) -> pd.DataFrame` — fit fresh clones on df,
    return merged wide frame `unique_id, ds, <model1>, [<model1>-lo-l, ...], <model2>...`
    (rename each model's yhat/lo/hi like cross_validation does).
  - `.cross_validate(df, h, n_windows=3, step_size=None, models=None, level=None)` —
    thin wrapper over `evaluation.backtest.cross_validation`.
  - `.compare(df, h, n_windows=3, metrics=("mae","rmse","smape"), seasonality=1,
    models=None) -> pd.DataFrame` — cross-validate, `evaluate(...)`, aggregate mean over
    series per (metric, model); return leaderboard: index=model name, columns=metrics,
    sorted ascending by `metrics[0]`.
  - Resolve string models via `get_model` at call time; accept instances too.
- `cli.py` — argparse, prog `forecast-os`, `main(argv=None) -> int`. Subcommands:
  - `models [--family FAMILY]` — print `list_models()` as a table
    (`DataFrame.to_string(index=False)`).
  - `forecast INPUT --h H [--model NAME] [--level L] [--output PATH]` — read CSV
    (`parse ds with pd.to_datetime` when non-numeric), default model `auto_ets`; write
    forecast CSV to --output or print. Import model modules lazily via
    `import forecast_os` (top-level import is enough post-integration).
  - `compare INPUT --h H [--models a,b,c] [--n-windows N] [--metrics m1,m2]
    [--output PATH]` — leaderboard via ForecastEngine.compare.
  - `simulate --s0 S0 --h H [--mu MU] [--sigma SIGMA] [--paths N] [--seed S]
    [--from-returns INPUT] [--output PATH]` — Monte Carlo summary table via
    `finance.montecarlo.MonteCarloSimulator`.
  - Errors (bad file, unknown model, contract violations) print `error: ...` to stderr
    and return exit code 2 — no tracebacks (catch ForecastOSError/ValueError/OSError).
- Adapters: `StatsForecastAdapter(model="AutoARIMA", season_length=1)` /
  `NeuralForecastAdapter(model="NHITS", max_steps=100)` — module-level import guard:
  `try: import statsforecast except ImportError: _HAS = False`; constructor raises
  ImportError with `pip install "forecast-os[nixtla]"` hint when missing. When present:
  translate fit/predict through the shared panel contract (statsforecast column names
  already match). Do NOT register in the global registry at import time; provide
  `register_adapters()` that registers whichever backends are installed (family
  "adapter").

**Tests:** engine.forecast columns/shape with two dummy models (locally defined and/or
locally registered names); compare leaderboard sorted by first metric and contains one
row per model; CLI via `main([...])` with tmp CSVs (tmp_path): models lists a registered
dummy; forecast writes a CSV with h rows per series and returns 0; forecast with a
missing file returns 2 and prints `error:`; compare prints a table; simulate prints
quantile summary and respects --seed reproducibility. Adapter: constructing without the
backend installed raises ImportError mentioning the extra; `register_adapters()` is a
no-op returning [] when nothing is installed.

### Task G: Core-layer tests

**Files:** Create `tests/test_types.py`, `tests/test_registry.py`,
`tests/test_metrics.py`, `tests/test_cross_validation.py`, `tests/test_datasets.py`.
Read the implementations in `src/forecast_os/core/`, `evaluation/`, `datasets/` first —
test actual behavior, and report (rather than work around) anything that looks wrong.

- test_types: missing column / empty / duplicate (uid, ds) / NaN y raise
  `DataContractError` with helpful messages; allow_missing passes NaN; unsorted input
  comes back sorted; string ds parsed to datetime; integer ds preserved; `to_panel`
  defaults; `infer_step` on daily/monthly/irregular datetime and on integer ds;
  `future_ds` continues both kinds.
- test_registry: register/get/list round-trip (use throwaway names prefixed `_test_`);
  duplicate name with a different class raises; same class re-register is a no-op;
  unknown family raises; non-forecaster class raises TypeError; get_model unknown name
  lists available; list_models(family=...) filters.
- test_metrics: hand-computed values for every metric (e.g. mae([1,2],[2,4])==1.5,
  rmse==sqrt(2.5), mape excludes zeros, smape 0/0 term==0, mase scale via seasonal naive
  on a known train series, pinball at q=0.5 == 0.5*MAE, coverage on known bounds);
  shape-mismatch and empty raise; evaluate(): correct rows (n_series×n_metrics), model
  column detection excludes `-lo-`/`-hi-`, mase without train_df raises, unknown metric
  raises.
- test_cross_validation: use a probe model (local PerSeriesForecaster subclass recording
  each fit's series lengths) to assert **no leakage** (for every window, train length ==
  total − h − offset) and correct cutoff spacing with default and custom step_size;
  output columns include per-model interval columns when level is passed; ds in each
  window's test rows are the h rows immediately after the cutoff; too-short series raise
  DataContractError; string model names resolve; duplicate names raise ValueError.
- test_datasets: generate_series shapes/determinism (same seed ⇒ identical frame,
  different seed ⇒ different values); seasonality=None has no cycle column effects;
  generate_returns garch validation raises on alpha+beta≥1; AirPassengers: 144 rows,
  first==112, last==432, monotone monthly ds, single unique_id.

### Task H (integration — main session, after A–G):
Wire `models/__init__.py`, `finance/__init__.py`, `uncertainty/__init__.py`,
`preprocessing/__init__.py`, `adapters/__init__.py`, and the package `__init__.py`
(import all model modules so registration happens on `import forecast_os`; re-export the
public API listed in the spec). Run the full suite + ruff; fix; commit.

### Task I (docs/CI/examples — after H): README, CONTRIBUTING, CHANGELOG, docs/,
examples/ (quickstart.py, financial_risk.py, model_comparison.py — runnable, no
matplotlib), `.github/workflows/ci.yml` (ruff + pytest, python 3.10–3.13,
ubuntu-latest, pip install -e ".[dev]").

### Task J (review + publish — after I): adversarial multi-agent review workflow;
fix confirmed findings; fresh-venv install + CLI smoke + `python -m build`; push.

## Self-review notes

- Spec coverage: every spec module maps to a task (A: baselines/ets/theta, B:
  arima/kalman, C: finance/*, D: ml/ensemble/auto/conformal, E: preprocessing/*, F:
  engine/cli/adapters, G: core tests, H: __init__ wiring, I: docs/CI, J: review+publish). ✔
- Deviation from the writing-plans template, documented: full implementation code is not
  inlined per step (≈4k LOC would duplicate the deliverable); instead each task pins
  exact signatures, equations, registry names, and named test cases, and implementers
  read the committed core sources. Commits are centralized (parallel agents + one git
  index). TDD order is mandated inside each task.
- Type consistency: interval columns `lo-{l}`/`hi-{l}` and CV renaming
  `{model}-lo-{l}` match core/base.py and evaluation/backtest.py as committed. ✔
