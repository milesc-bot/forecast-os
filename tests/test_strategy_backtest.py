"""Tests for finance/backtest.py: forecast-driven long/flat StrategyBacktester."""

from __future__ import annotations

import numpy as np
import pytest

from forecast_os.core.base import PerSeriesForecaster
from forecast_os.core.types import ID_COL, to_panel
from forecast_os.datasets.synthetic import generate_returns
from forecast_os.finance.backtest import BacktestResult, StrategyBacktester

SUMMARY_COLS = [
    ID_COL,
    "total_return",
    "annualized_return",
    "sharpe",
    "sortino",
    "max_drawdown",
    "hit_rate",
    "n_trades",
    "exposure",
    "annualized_vol",
    "calmar",
    "var_95",
    "cvar_95",
]
FRAME_COLS = [ID_COL, "ds", "y", "yhat", "position", "strategy_return", "equity"]
SIZED_FRAME_COLS = [ID_COL, "ds", "y", "yhat", "sigma", "position", "strategy_return", "equity"]


class MeanForecaster(PerSeriesForecaster):
    """Forecast the in-sample mean (a positive constant on drifting returns)."""

    def _fit_series(self, y):
        return {"mean": float(np.mean(y))}

    def _predict_series(self, state, h):
        return np.full(h, state["mean"])


@pytest.fixture(scope="module")
def drift_df():
    """One asset with strongly positive drift: mean forecast is always > 0."""
    return generate_returns(n_series=1, length=200, mu=0.005, sigma=0.005, seed=1)


@pytest.fixture(scope="module")
def drift_result(drift_df):
    return StrategyBacktester(MeanForecaster()).run(drift_df, test_size=60)


def test_result_types_and_columns(drift_result):
    assert isinstance(drift_result, BacktestResult)
    assert list(drift_result.summary.columns) == SUMMARY_COLS
    assert list(drift_result.frame.columns) == FRAME_COLS
    assert len(drift_result.summary) == 1
    assert len(drift_result.frame) == 60


def test_always_long_equals_buy_and_hold(drift_df, drift_result):
    last60 = drift_df["y"].to_numpy()[-60:]
    buy_hold = float(np.prod(1.0 + last60) - 1.0)
    row = drift_result.summary.iloc[0]
    assert (drift_result.frame["position"] == 1.0).all()
    assert np.isclose(row["total_return"], buy_hold, rtol=1e-10)
    assert row["exposure"] == 1.0
    assert row["n_trades"] == 1  # single entry trade at the start
    np.testing.assert_allclose(drift_result.frame["y"], last60)
    np.testing.assert_allclose(drift_result.frame["strategy_return"], last60)


def test_equity_is_cumprod_of_strategy_returns(drift_result):
    g = drift_result.frame
    np.testing.assert_allclose(
        g["equity"].to_numpy(), np.cumprod(1.0 + g["strategy_return"].to_numpy())
    )


def test_costs_reduce_total_return(drift_df, drift_result):
    costly = StrategyBacktester(MeanForecaster(), cost_bps=50.0).run(drift_df, test_size=60)
    assert (
        costly.summary["total_return"].iloc[0] < drift_result.summary["total_return"].iloc[0]
    )


def test_high_threshold_stays_flat(drift_df):
    res = StrategyBacktester(MeanForecaster(), threshold=10.0).run(drift_df, test_size=60)
    row = res.summary.iloc[0]
    assert (res.frame["position"] == 0.0).all()
    assert (res.frame["strategy_return"] == 0.0).all()
    assert row["total_return"] == 0.0
    assert row["exposure"] == 0.0
    assert row["n_trades"] == 0
    assert row["max_drawdown"] == 0.0


def test_multi_series_summary_one_row_per_uid():
    df = generate_returns(n_series=2, length=150, mu=0.003, sigma=0.01, seed=2)
    res = StrategyBacktester(MeanForecaster()).run(df, test_size=40)
    assert len(res.summary) == 2
    assert set(res.summary[ID_COL]) == {"asset-0", "asset-1"}
    assert len(res.frame) == 80
    assert (res.frame.groupby(ID_COL).size() == 40).all()


class ParityFlipForecaster(PerSeriesForecaster):
    """h=1 forecast alternates sign with training length: +0.01 if even, -0.01 if odd."""

    def _fit_series(self, y):
        return {"sign": 1.0 if len(y) % 2 == 0 else -1.0}

    def _predict_series(self, state, h):
        return np.full(h, 0.01 * state["sign"])


def test_position_flips_hand_computed_costs_trades_exposure():
    """Hand-computed flip sequence: cost is debited on every entry AND exit.

    n=12 returns, test_size=6, step_size=1: the walk-forward training lengths
    for the 6 test periods are 6,7,8,9,10,11, so the parity forecaster goes
    long, flat, long, flat, long, flat. Every period is a position change
    (|dpos|=1, starting flat), so each period is charged cost_bps/1e4 = 0.01.
    """
    y = np.array(
        [0.010, -0.020, 0.015, 0.005, -0.010, 0.020,  # train-only head
         0.030, -0.040, 0.050, -0.060, 0.070, -0.080]  # walk-forward test tail
    )
    df = to_panel(y, unique_id="asset-0")
    res = StrategyBacktester(ParityFlipForecaster(), cost_bps=100.0).run(
        df, test_size=6, step_size=1
    )

    frame = res.frame
    assert len(frame) == 6
    np.testing.assert_allclose(
        frame["yhat"].to_numpy(), [0.01, -0.01, 0.01, -0.01, 0.01, -0.01]
    )
    np.testing.assert_allclose(
        frame["position"].to_numpy(), [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    )
    # long periods earn y_t minus the 0.01 entry cost; flat periods pay the
    # 0.01 exit cost and earn nothing
    expected_strat_ret = np.array([0.020, -0.010, 0.040, -0.010, 0.060, -0.010])
    np.testing.assert_allclose(
        frame["strategy_return"].to_numpy(), expected_strat_ret, atol=1e-12
    )
    np.testing.assert_allclose(
        frame["equity"].to_numpy(), np.cumprod(1.0 + expected_strat_ret), atol=1e-12
    )

    row = res.summary.iloc[0]
    assert row["n_trades"] == 6  # one position change every period
    assert row["exposure"] == 0.5  # long half of the periods
    assert np.isclose(
        row["total_return"], float(np.prod(1.0 + expected_strat_ret) - 1.0)
    )


def test_params_stored_as_attributes():
    model = MeanForecaster()
    bt = StrategyBacktester(model, threshold=0.5, cost_bps=10.0)
    assert bt.model is model
    assert bt.threshold == 0.5
    assert bt.cost_bps == 10.0


# ---------------------------------------------------------------------------
# sizing rules (proportional / kelly) and the risk report
# ---------------------------------------------------------------------------

RISK_COLS = ["annualized_vol", "calmar", "var_95", "cvar_95"]


class ScriptedForecaster(PerSeriesForecaster):
    """Deterministic h=1 forecaster with fully controllable yhat and sigma.

    Walk-forward window ``i`` trains on ``offset + i`` rows, so the step index
    is recovered as ``len(y) - offset`` and used to look up ``yhats[i]`` /
    ``sigmas[i]``.
    """

    def __init__(self, yhats, sigmas, offset):
        self.yhats = yhats
        self.sigmas = sigmas
        self.offset = offset

    def _fit_series(self, y):
        return {"idx": len(y) - self.offset}

    def _predict_series(self, state, h):
        return np.full(h, self.yhats[state["idx"]])

    def _predict_sigma(self, state, h):
        return np.full(h, self.sigmas[state["idx"]])


def scripted_run(yhats, sigmas, y_test=None, **bt_kwargs):
    """Backtest a ScriptedForecaster over exactly ``len(yhats)`` test steps."""
    k = len(yhats)
    head = [0.001, -0.002, 0.003, -0.001, 0.002, -0.003, 0.001, -0.002]
    y_test = [0.01] * k if y_test is None else list(y_test)
    y = np.array(head + y_test)
    model = ScriptedForecaster(tuple(yhats), tuple(sigmas), offset=len(head))
    return StrategyBacktester(model, **bt_kwargs).run(to_panel(y), test_size=k, step_size=1)


def test_binary_summary_gains_risk_columns_finite_and_ordered(drift_result):
    """Existing binary fixture: new risk columns present, finite, cvar >= var >= 0."""
    row = drift_result.summary.iloc[0]
    for col in RISK_COLS:
        assert np.isfinite(row[col]), col
    assert row["var_95"] >= 0.0
    assert row["cvar_95"] >= row["var_95"]


def test_binary_risk_columns_match_metrics_functions(drift_result):
    from forecast_os.finance.metrics import (
        annualized_vol,
        calmar_ratio,
        conditional_var,
        value_at_risk,
    )

    r = drift_result.frame["strategy_return"].to_numpy()
    row = drift_result.summary.iloc[0]
    assert np.isclose(row["annualized_vol"], annualized_vol(r))
    assert np.isclose(row["calmar"], calmar_ratio(r))
    assert np.isclose(row["var_95"], value_at_risk(r, level=0.95))
    assert np.isclose(row["cvar_95"], conditional_var(r, level=0.95))


def test_binary_frame_has_no_sigma_column(drift_result):
    assert "sigma" not in drift_result.frame.columns


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"sizing": "martingale"},
        {"sizing": ""},
        {"level": 0},
        {"level": 100},
        {"level": -5},
        {"kelly_fraction": 0.0},
        {"kelly_fraction": -0.5},
        {"kelly_fraction": float("nan")},
        {"max_leverage": 0.0},
        {"max_leverage": -1.0},
    ],
)
def test_init_rejects_bad_sizing_params(bad_kwargs):
    with pytest.raises(ValueError):
        StrategyBacktester(MeanForecaster(), **bad_kwargs)


def test_sizing_params_stored_as_attributes():
    bt = StrategyBacktester(
        MeanForecaster(), sizing="kelly", level=90, kelly_fraction=0.25, max_leverage=2.0
    )
    assert bt.sizing == "kelly"
    assert bt.level == 90
    assert bt.kelly_fraction == 0.25
    assert bt.max_leverage == 2.0


def test_proportional_positions_bounded_and_monotone_in_yhat():
    """For fixed sigma, proportional positions are in [0, 1] and monotone in yhat."""
    from scipy import stats

    yhats = (-0.05, -0.01, -0.002, 0.0, 0.002, 0.01, 0.05, 0.2)
    sigmas = (0.01,) * len(yhats)
    res = scripted_run(yhats, sigmas, sizing="proportional")

    assert list(res.frame.columns) == SIZED_FRAME_COLS
    pos = res.frame["position"].to_numpy()
    assert np.all(pos >= 0.0) and np.all(pos <= 1.0)
    assert np.all(np.diff(pos) >= 0.0)  # monotone nondecreasing in yhat
    assert np.all(np.diff(pos[3:7]) > 0.0)  # strictly increasing off the clip bounds

    p_up = stats.norm.sf((0.0 - np.asarray(yhats)) / 0.01)
    expected = np.clip(2.0 * (p_up - 0.5), 0.0, 1.0)
    np.testing.assert_allclose(pos, expected, atol=1e-10)
    # sigma recovered from the (lo, hi) interval round-trips exactly
    np.testing.assert_allclose(res.frame["sigma"].to_numpy(), sigmas, atol=1e-12)


def test_kelly_interior_value_and_max_leverage_clip():
    """Hand-computed kelly step: 0.5 * 0.004 / 0.1**2 = 0.2; big edges clip at max_leverage."""
    yhats = (0.004, 0.05, -0.01, 0.008)
    sigmas = (0.1, 0.05, 0.1, 0.2)
    res = scripted_run(yhats, sigmas, sizing="kelly", kelly_fraction=0.5, max_leverage=1.5)
    pos = res.frame["position"].to_numpy()
    # interior: kelly_fraction * (yhat - threshold) / sigma**2
    np.testing.assert_allclose(pos, [0.5 * 0.004 / 0.01, 1.5, 0.0, 0.5 * 0.008 / 0.04], atol=1e-10)
    assert np.all(pos <= 1.5)


@pytest.mark.parametrize("sizing", ["proportional", "kelly"])
def test_zero_sigma_falls_back_to_binary(sizing):
    yhats = (0.01, -0.01, 0.02, 0.005)
    sigmas = (0.0, 0.0, 0.02, 0.0)
    res = scripted_run(yhats, sigmas, sizing=sizing, kelly_fraction=0.5, max_leverage=1.0)
    pos = res.frame["position"].to_numpy()
    # steps 0, 1, 3 have sigma == 0 -> binary rule; step 2 is sized normally
    assert pos[0] == 1.0
    assert pos[1] == 0.0
    assert pos[3] == 1.0
    assert 0.0 < pos[2] <= 1.0


def test_fractional_resize_cost_accounting_hand_computed():
    """Position path 0 -> 0.4 -> 1.0 -> 0.0 charges cost_bps/1e4 * |dpos| each step."""
    yhats = (0.008, 0.02, -0.01)
    sigmas = (0.1, 0.1, 0.1)
    y_test = [0.05, -0.02, 0.03]
    res = scripted_run(
        yhats,
        sigmas,
        y_test=y_test,
        sizing="kelly",
        kelly_fraction=0.5,
        max_leverage=1.0,
        cost_bps=100.0,
    )
    pos = res.frame["position"].to_numpy()
    np.testing.assert_allclose(pos, [0.4, 1.0, 0.0], atol=1e-10)
    # |dpos| = [0.4, 0.6, 1.0] at 100 bps -> per-step costs 0.004, 0.006, 0.010
    expected = np.array([0.4 * 0.05 - 0.004, 1.0 * -0.02 - 0.006, 0.0 * 0.03 - 0.010])
    np.testing.assert_allclose(res.frame["strategy_return"].to_numpy(), expected, atol=1e-10)
    np.testing.assert_allclose(
        res.frame["equity"].to_numpy(), np.cumprod(1.0 + expected), atol=1e-10
    )
    assert res.summary.iloc[0]["n_trades"] == 3


class AR1Forecaster(PerSeriesForecaster):
    """Least-squares AR(1) with residual-based default sigma."""

    min_train_size = 3

    def _fit_series(self, y):
        y0, y1 = y[:-1], y[1:]
        phi = float(np.dot(y0, y1) / np.dot(y0, y0))
        fitted = np.concatenate([[np.nan], phi * y[:-1]])
        return {"phi": phi, "last": float(y[-1]), "fitted": fitted}

    def _predict_series(self, state, h):
        return state["phi"] ** np.arange(1, h + 1) * state["last"]


def test_proportional_on_predictable_ar_panel_partial_exposure():
    """Strongly autocorrelated returns: proportional sizing is partially invested."""
    rng = np.random.default_rng(7)
    n = 300
    y = np.zeros(n)
    eps = 0.002 * rng.standard_normal(n)
    for t in range(1, n):
        y[t] = 0.9 * y[t - 1] + eps[t]
    df = to_panel(y, unique_id="asset-0")

    res = StrategyBacktester(AR1Forecaster(), sizing="proportional").run(df, test_size=60)
    pos = res.frame["position"].to_numpy()
    assert np.all(pos >= 0.0) and np.all(pos <= 1.0)
    row = res.summary.iloc[0]
    assert 0.0 < row["exposure"] < 1.0
    assert np.isfinite(row["sharpe"])
    assert np.isfinite(res.frame["sigma"]).all()


def test_kelly_sigma_fallback_respects_max_leverage():
    """Degenerate-sigma steps fall back to binary capped at max_leverage."""
    result = scripted_run(
        yhats=[0.01, 0.01, -0.01],
        sigmas=[0.0, 0.02, 0.0],
        sizing="kelly",
        level=80,
        kelly_fraction=0.5,
        max_leverage=0.4,
    )
    pos = result.frame["position"].to_numpy()
    assert pos[0] == pytest.approx(0.4)  # binary 1.0 capped at max_leverage
    assert pos[1] == pytest.approx(min(0.5 * 0.01 / 0.02**2, 0.4))
    assert pos[2] == 0.0
    assert (pos <= 0.4 + 1e-12).all()


class ConstForecaster(PerSeriesForecaster):
    """Always forecast a fixed positive return."""

    def _fit_series(self, y):
        return {"c": 0.05}

    def _predict_series(self, state, h):
        return np.full(h, 0.05)


def _monthly_panel():
    import pandas as pd

    return pd.DataFrame(
        {
            ID_COL: "asset-0",
            "ds": pd.date_range("2015-01-31", periods=120, freq="ME"),
            "y": 0.05,
        }
    )


def test_annualization_follows_the_panel_frequency():
    """Annualized summary metrics must use the panel's own periods-per-year.

    Regression: every summary metric was called with metrics.py's ``periods``
    default of 252, and ``StrategyBacktester`` exposed no override. A monthly
    panel of constant +5% returns therefore reported annualized_return =
    218625.78 (1.05**252 - 1) instead of the frequency-correct 0.79586
    (1.05**12 - 1), with sharpe/sortino/annualized_vol/calmar inflated by
    sqrt(252/12) alongside. Nothing documented a daily-bar assumption, and
    there was no workaround short of recomputing from ``result.frame``.
    """
    res = StrategyBacktester(ConstForecaster()).run(_monthly_panel(), test_size=24)
    row = res.summary.iloc[0]
    assert row["annualized_return"] == pytest.approx(1.05**12 - 1.0)
    assert res.periods == 12
    # frequency-free metrics are unaffected
    assert row["total_return"] == pytest.approx(1.05**24 - 1.0)


def test_annualization_periods_can_be_overridden():
    """An explicit ``periods`` wins over inference."""
    res = StrategyBacktester(ConstForecaster(), periods=4).run(_monthly_panel(), test_size=24)
    assert res.periods == 4
    assert res.summary.iloc[0]["annualized_return"] == pytest.approx(1.05**4 - 1.0)


def test_annualization_defaults_to_252_on_daily_panels(drift_result):
    """Business/calendar-daily panels keep the historical 252 convention."""
    assert drift_result.periods == 252


def test_periods_validated():
    with pytest.raises(ValueError):
        StrategyBacktester(ConstForecaster(), periods=0)
    with pytest.raises(ValueError):
        StrategyBacktester(ConstForecaster(), periods=2.5)


def _daily_panel(ds):
    import pandas as pd

    return pd.DataFrame({ID_COL: "asset-0", "ds": ds, "y": 0.01})


def test_run_accepts_contract_legal_object_dtype_ds():
    """Backward compat: ISO-string / ``datetime.date`` ``ds`` must still run.

    Regression: the annualization fix inferred the periods-per-year from the
    caller's *raw* frame as run()'s first statement, before
    ``cross_validation`` -> ``validate_panel`` had a chance to normalise ``ds``.
    ``validate_panel`` explicitly accepts an object-dtype ``ds`` and coerces it
    with ``pd.to_datetime`` (exactly what ``pd.read_csv`` without
    ``parse_dates`` produces), so both panels below completed at v0.9.0; after
    the fix they raised ``TypeError: operation 'sub' not supported for dtype
    'str'`` and ``TypeError: float() argument must be ... not 'Timedelta'``.
    Periods must be inferred from the validated frame.
    """
    import pandas as pd

    idx = pd.date_range("2020-01-01", periods=120, freq="D")
    for ds in (
        idx.strftime("%Y-%m-%d"),  # ISO strings, object dtype
        [d.date() for d in idx],  # datetime.date objects
        [d.to_pydatetime() for d in idx],  # datetime.datetime objects
    ):
        res = StrategyBacktester(ConstForecaster()).run(_daily_panel(ds), test_size=12)
        assert res.periods == 252
        assert len(res.frame) == 12
        assert isinstance(res, BacktestResult)


def test_malformed_panels_still_raise_data_contract_error():
    """Backward compat: contract violations keep their actionable error.

    Regression: inferring the periods-per-year before validation meant a
    malformed panel hit ``df[ID_COL]`` / ``groupby`` first and surfaced a bare
    ``KeyError`` / ``IndexError`` / ``AttributeError`` instead of the package's
    ``DataContractError``.
    """
    import pandas as pd

    from forecast_os.core.exceptions import DataContractError

    idx = pd.date_range("2020-01-01", periods=120, freq="D")
    cases = [
        pd.DataFrame({"ds": idx, "y": 0.01}),  # missing unique_id
        pd.DataFrame({ID_COL: [], "ds": [], "y": []}),  # empty
        pd.DataFrame({ID_COL: [None] * 120, "ds": idx, "y": 0.01}),  # null ids
        {"unique_id": "a"},  # not a DataFrame at all
    ]
    for case in cases:
        with pytest.raises(DataContractError):
            StrategyBacktester(ConstForecaster()).run(case, test_size=12)


def test_inferred_periods_account_for_step_size():
    """Inferred annualization must match the *trade* cadence, not the panel's.

    Regression: ``periods`` was inferred from the panel's bar spacing alone, but
    the realized return series is ``step_size`` bars apart. On a business-daily
    panel with ``step_size=5`` the 40 realized returns span 273 calendar days,
    yet ``periods`` was reported as 252 — a 5.8x overstatement of
    annualized_return (0.43 vs the honest 0.07), under a ``BacktestResult.periods``
    documented as "the periods-per-year actually used to annualize".
    """
    import pandas as pd

    from forecast_os.finance.metrics import annualized_return as ann

    df = _daily_panel(pd.bdate_range("2020-01-01", periods=300))
    res = StrategyBacktester(ConstForecaster()).run(df, test_size=40, step_size=5)
    assert res.periods == 50  # 252 business days / 5-day rebalance
    strat = res.frame["strategy_return"].to_numpy()
    assert res.summary.iloc[0]["annualized_return"] == pytest.approx(ann(strat, periods=50))
    # step_size=1 is unchanged
    assert StrategyBacktester(ConstForecaster()).run(df, test_size=40).periods == 252


def test_inferred_periods_honor_the_frequency_multiple():
    """Multi-unit frequencies must annualize off the *step*, not the base unit.

    Regression: inference looked the inferred frequency string up in a table
    keyed by bare unit roots, so only single-unit steps ever matched. A 4-hour
    bar panel — which ``pd.infer_freq`` resolves cleanly to ``"4h"``, so neither
    numeric nor irregular ``ds`` — missed ``"h"`` and silently took the 252
    fallback instead of 24*252/4 = 1512, understating annualized_vol by 59%
    (and 15-minute bars by ~90%). ``"2D"``/``"2W"`` were wrong the other way:
    they inherited the *full* daily/weekly rate rather than half of it. The
    multiple is available as ``to_offset(freq).n``, and periods-per-year is
    simply the base unit's rate divided by it.
    """
    import pandas as pd

    from forecast_os.finance.backtest import _infer_periods

    def panel(freq):
        return pd.DataFrame(
            {ID_COL: "asset-0", "ds": pd.date_range("2020-01-01", periods=60, freq=freq), "y": 0.01}
        )

    assert _infer_periods(panel("4h")) == 1512  # 24 * 252 / 4
    assert _infer_periods(panel("2h")) == 3024
    assert _infer_periods(panel("2D")) == 126  # 252 / 2
    assert _infer_periods(panel("2W")) == 26  # 52 / 2, anchor suffix ignored
    assert _infer_periods(panel("2ME")) == 6
    # single-unit steps and the documented fallbacks are unchanged
    assert _infer_periods(panel("h")) == 6048
    assert _infer_periods(panel("D")) == 252
    assert _infer_periods(panel("ME")) == 12
    assert _infer_periods(panel("W")) == 52
    # Sub-hourly and business-anchored units are in the table too: falling back
    # to 252 understated a 15-minute panel's annualized vol ~10x and OVERSTATED
    # a business-month-end panel's Sharpe ~4.6x.
    assert _infer_periods(panel("15min")) == 24 * 252 * 4  # 24*252*60 / 15
    assert _infer_periods(panel("min")) == 24 * 252 * 60
    assert _infer_periods(panel("BME")) == 12
    assert _infer_periods(panel("BQE")) == 4
    assert _infer_periods(panel("BYE")) == 1


def test_intraday_panel_reports_the_multiplied_periods_end_to_end():
    """``BacktestResult.periods`` must carry the multiple-aware rate."""
    import pandas as pd

    df = _daily_panel(pd.date_range("2020-01-01", periods=200, freq="4h"))
    res = StrategyBacktester(ConstForecaster()).run(df, test_size=12)
    assert res.periods == 1512
    assert res.summary.iloc[0]["annualized_vol"] == pytest.approx(0.0, abs=1e-9)
