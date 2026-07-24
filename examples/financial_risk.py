"""Financial modeling tour: GARCH volatility, Monte Carlo scenarios,
regime detection, risk metrics, and a forecast-driven strategy backtest.

Run:  python examples/financial_risk.py
"""

import numpy as np

import forecast_os as fos
from forecast_os.finance import (
    GARCH11,
    MarkovRegimeSwitching,
    MonteCarloSimulator,
    StrategyBacktester,
    max_drawdown,
    sharpe_ratio,
    value_at_risk,
)

# Simulated daily returns with GARCH volatility clustering
panel = fos.generate_returns(length=1500, mu=0.0003, garch=(2e-6, 0.08, 0.88), seed=11)
returns = panel["y"].to_numpy()

print("=== Risk metrics ===")
print(f"annualized Sharpe : {sharpe_ratio(returns):.2f}")
print(f"max drawdown      : {max_drawdown(returns):.1%}")
print(f"1-day 95% VaR     : {value_at_risk(returns, level=0.95):.2%}")

print("\n=== GARCH(1,1) conditional volatility ===")
garch = GARCH11().fit(returns)
print(f"alpha + beta (persistence): {garch.alpha_ + garch.beta_:.3f}")
print(f"next-5-day vol forecast   : {np.round(garch.forecast_volatility(5), 5)}")

print("\n=== Monte Carlo: 1-year price scenarios from $100 ===")
mc = MonteCarloSimulator.from_returns(returns, seed=7)
fan = mc.summary(s0=100.0, h=252, n_paths=2000).iloc[[0, 62, 125, 251]]
print(fan.round(2).to_string(index=False))

print("\n=== Bull/bear regime detection ===")
reg = MarkovRegimeSwitching(seed=3).fit(returns)
print(f"state means (bear→bull): {np.round(reg.means_, 5)}")
print(f"state stds             : {np.round(reg.stds_, 5)}")
print(f"P(bull) today          : {reg.smoothed_probs_[-1, 1]:.2f}")

print("\n=== Strategy backtest: ridge forecaster, long/flat, 1bp costs ===")
bt = StrategyBacktester(fos.get_model("ridge_lag", lags=5), cost_bps=1.0)
result = bt.run(panel, test_size=120)
print(result.summary.round(3).to_string(index=False))

print("\n=== Same forecaster, uncertainty-aware sizing (proportional) ===")
bt_prop = StrategyBacktester(
    fos.get_model("ridge_lag", lags=5), cost_bps=1.0, sizing="proportional", level=80
)
result_prop = bt_prop.run(panel, test_size=120)
print(result_prop.summary.round(3).to_string(index=False))
