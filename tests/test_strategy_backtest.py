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
]
FRAME_COLS = [ID_COL, "ds", "y", "yhat", "position", "strategy_return", "equity"]


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
