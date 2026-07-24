"""Whole-engine A/B benchmark on sample GTM data.

Part 1 is the model bakeoff — every model in the zoo forecasts the same
sample bookings panel under walk-forward cross-validation, ranked by accuracy
and interval calibration (the champion/challenger "A/B test").

Part 2 is a full-engine health check: reconciliation, quota, conformal
calibration, exogenous drivers, finance, snapshots, connectors, and
persistence each run and report PASS/FAIL.

Run:  python examples/engine_benchmark.py
Exit code 0 iff every subsystem passes.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import forecast_os as fos

SEED = 20260724
M = 12  # monthly seasonality


def sample_panel() -> pd.DataFrame:
    """6 hierarchical team/rep bookings series, 48 months, trend + seasonality."""
    return fos.generate_series(
        n_series=6, length=48, freq="MS", level=400.0, trend=4.0,
        seasonality=M, season_amp=60.0, noise=12.0, seed=SEED,
    ).assign(
        unique_id=lambda d: d["unique_id"].map(dict(zip(
            [f"series-{i}" for i in range(6)],
            ["west/alice", "west/bob", "west/cara", "east/dan", "east/erin", "east/fay"],
        )))
    )


# --------------------------------------------------------------------------- #
# Part 1 — the model A/B bakeoff                                              #
# --------------------------------------------------------------------------- #


def bakeoff(panel: pd.DataFrame) -> pd.DataFrame:
    engine = fos.ForecastEngine()
    board = engine.compare(
        panel, h=6, n_windows=3, seasonality=M, level=[80],
        metrics=["mase", "rmse", "coverage", "pct_bias"],
        models=[
            "naive", "seasonal_naive", "drift", "ses", "kalman", "auto_arima",
            fos.get_model("theta", season_length=M),
            fos.get_model("holt_winters", season_length=M),
            fos.get_model("auto_ets", season_length=M),
            fos.get_model("ridge_lag", lags=12, season_length=M),
            "ensemble",
        ],
    )
    return board


# --------------------------------------------------------------------------- #
# Part 2 — whole-engine health checks                                        #
# --------------------------------------------------------------------------- #


def check(name, fn):
    try:
        detail = fn()
        print(f"  [PASS] {name:<28} {detail}")
        return True
    except Exception as exc:  # noqa: BLE001 - a health check reports, never crashes
        print(f"  [FAIL] {name:<28} {type(exc).__name__}: {exc}")
        return False


def chk_reconciliation(panel):
    m = fos.get_model("reconciled", model="auto_ets", method="mint_ols").fit(panel)
    fc = m.predict(6, level=[80])
    tot = fc[fc["unique_id"] == "total"].reset_index(drop=True)
    reps = fc[fc["unique_id"].str.contains("/")].groupby("ds")["yhat"].sum().to_numpy()
    gap = float(np.max(np.abs(tot["yhat"].to_numpy() - reps)))
    assert gap < 1e-6, f"incoherent by {gap}"
    return f"reps sum to total within {gap:.2e}"


def chk_quota(panel):
    m = fos.get_model("reconciled", model="auto_ets").fit(panel)
    fc = m.predict(6, level=[80])
    att = fos.gtm.attainment_probability(
        fc[fc["unique_id"] == "total"], quota={"total": 15_000.0}, level=80
    )
    p = float(att["p_attain"].iloc[0])
    assert 0.0 <= p <= 1.0
    return f"P(total >= quota) = {p:.2f}"


def chk_conformal(panel):
    per = len(panel) // panel["unique_id"].nunique()
    train = panel.groupby("unique_id").head(per - 6)
    test = panel.groupby("unique_id").tail(6)
    conf = fos.ConformalForecaster(model=fos.get_model("theta", season_length=M)).fit(train)
    pred = conf.predict(6, level=[80]).merge(
        test[["unique_id", "ds", "y"]], on=["unique_id", "ds"]
    )
    cov = float(((pred["y"] >= pred["lo-80"]) & (pred["y"] <= pred["hi-80"])).mean())
    assert 0.4 <= cov <= 1.0, f"coverage {cov}"
    return f"80% interval holdout coverage = {cov:.0%}"


def chk_exog_arima():
    rng = np.random.default_rng(SEED)
    n = 80
    x = rng.normal(size=n)
    y = 5 * x + np.cumsum(rng.normal(0, 0.3, n)) + 100
    df = pd.DataFrame({"unique_id": "s", "ds": np.arange(n), "y": y, "x": x})
    m = fos.get_model("arima", order=(1, 0, 0)).fit(df)
    hi = m.predict(3, X_df=pd.DataFrame({"unique_id": "s", "ds": range(n, n + 3), "x": 2.0})
                   )["yhat"].mean()
    lo = m.predict(3, X_df=pd.DataFrame({"unique_id": "s", "ds": range(n, n + 3), "x": -2.0})
                   )["yhat"].mean()
    assert hi - lo > 5, f"driver delta {hi - lo}"
    return f"ARIMA responds to driver (delta {hi - lo:.1f})"


def chk_finance():
    from forecast_os.finance import GARCH11, StrategyBacktester
    panel = fos.generate_returns(length=1000, garch=(2e-6, 0.08, 0.88), seed=SEED)
    g = GARCH11().fit(panel["y"].to_numpy())
    persistence = g.alpha_ + g.beta_
    assert 0.7 <= persistence <= 0.999, f"persistence {persistence}"
    bt = StrategyBacktester(fos.get_model("ridge_lag", lags=5), cost_bps=1.0)
    res = bt.run(panel, test_size=120)
    assert np.isfinite(res.summary["sharpe"].iloc[0])
    return f"GARCH persistence {persistence:.2f}, backtest Sharpe finite"


def chk_snapshots(panel):
    per = len(panel) // panel["unique_id"].nunique()
    store = fos.snapshots.SnapshotStore(Path(tempfile.mkdtemp()))
    known = panel.groupby("unique_id").head(per - 6)
    store.snapshot(known, as_of="2026-07-06", kind="panel")
    fc = fos.get_model("auto_ets", season_length=M).fit(known).predict(6, level=[80])
    store.snapshot(fc, as_of="2026-07-06", kind="forecast", label="commit")
    audit = fos.snapshots.forecast_vs_actual(store.history(kind="forecast"), panel)
    assert len(audit) > 0
    return f"{len(store.as_of_dates('panel'))} snapshot(s), {len(audit)} forecast/actual pairs"


def chk_connectors():
    rng = np.random.default_rng(SEED)
    hubspot = pd.DataFrame({
        "closedate": rng.choice(pd.date_range("2024-01-01", "2025-12-31"), 200),
        "amount": rng.lognormal(9, 0.4, 200).round(2),
        "dealstage": rng.choice(["closedwon", "closedlost"], 200, p=[0.6, 0.4]),
    })
    p = fos.connectors.apply_mapping(hubspot, "hubspot_deals")
    fos.validate_panel(p)
    return f"hubspot_deals -> {len(p)}-row panel, ${p['y'].sum():,.0f} won"


def chk_persistence(panel):
    from forecast_os.core.base import load
    m = fos.get_model("auto_ets", season_length=M).fit(panel)
    path = Path(tempfile.mkdtemp()) / "m.pkl"
    m.save(path)
    reloaded = load(path)
    a = m.predict(3, level=[80])["yhat"].to_numpy()
    b = reloaded.predict(3, level=[80])["yhat"].to_numpy()
    assert np.allclose(a, b)
    return "save/load round-trips identically"


def main() -> int:
    panel = sample_panel()
    print(f"Sample data: {panel['unique_id'].nunique()} series x "
          f"{len(panel) // panel['unique_id'].nunique()} months\n")

    print("=" * 66)
    print("PART 1 — MODEL A/B BAKEOFF  (walk-forward CV, 3 windows, h=6)")
    print("=" * 66)
    board = bakeoff(panel)
    print(board.round(3).to_string())
    champ = board.index[0]
    naive_key = next((k for k in board.index if k.lower() == "naive"), None)
    naive_mase = board.loc[naive_key, "mase"] if naive_key else float("nan")
    lift = (naive_mase - board.loc[champ, "mase"]) / naive_mase if naive_mase else float("nan")
    print(f"\n  champion: {champ}  (MASE {board.loc[champ, 'mase']:.3f}, "
          f"{lift:.0%} better than naive, "
          f"80% coverage {board.loc[champ, 'coverage-80']:.0%})")

    print("\n" + "=" * 66)
    print("PART 2 — WHOLE-ENGINE HEALTH CHECK")
    print("=" * 66)
    checks = [
        ("hierarchical reconciliation", lambda: chk_reconciliation(panel)),
        ("quota attainment", lambda: chk_quota(panel)),
        ("conformal calibration", lambda: chk_conformal(panel)),
        ("exogenous ARIMA", chk_exog_arima),
        ("finance (GARCH + backtest)", chk_finance),
        ("snapshot store", lambda: chk_snapshots(panel)),
        ("connectors (HubSpot)", chk_connectors),
        ("model persistence", lambda: chk_persistence(panel)),
    ]
    passed = sum(check(name, fn) for name, fn in checks)

    print("\n" + "=" * 66)
    ok = passed == len(checks)
    verdict = "ALL SYSTEMS GO" if ok else f"{len(checks) - passed} SUBSYSTEM(S) FAILED"
    print(f"  {passed}/{len(checks)} subsystems passed  ->  {verdict}")
    print("=" * 66)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
