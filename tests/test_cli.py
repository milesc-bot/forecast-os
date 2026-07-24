"""CLI tests: drive ``forecast_os.cli.main(argv)`` with tmp CSVs.

Self-sufficient: registers local dummy forecasters under ``_test_``-prefixed
names at run time. The ``simulate`` command is tested against a minimal fake
``forecast_os.finance.montecarlo`` module when the real one is not built yet.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from forecast_os.cli import main
from forecast_os.core.base import PerSeriesForecaster
from forecast_os.core.registry import register


class _CliNaive(PerSeriesForecaster):
    """Last-value dummy used to exercise the CLI."""

    alias = "_test_cli_naive"

    def _fit_series(self, y):
        return {"last": float(y[-1])}

    def _predict_series(self, state, h):
        return np.full(h, state["last"])


class _CliMean(PerSeriesForecaster):
    """Mean-value dummy used to exercise the CLI."""

    alias = "_test_cli_mean"

    def _fit_series(self, y):
        return {"mean": float(np.mean(y))}

    def _predict_series(self, state, h):
        return np.full(h, state["mean"])


class _CliSeasonal(PerSeriesForecaster):
    """Season-repeating dummy proving typed --param values reach the model."""

    alias = "_test_cli_seasonal"

    def __init__(self, season_length: int = 1):
        self.season_length = season_length
        self.min_train_size = season_length

    def _fit_series(self, y):
        # slicing with a str season_length would raise, so success proves the
        # CLI coerced the --param value to int
        return {"tail": np.asarray(y[-self.season_length :], dtype=float)}

    def _predict_series(self, state, h):
        return np.resize(state["tail"], h)


@pytest.fixture(autouse=True)
def _register_dummies():
    # re-registering the same class is a no-op, so autouse is safe
    register("_test_cli_naive", family="baseline")(_CliNaive)
    register("_test_cli_mean", family="baseline")(_CliMean)
    register("_test_cli_seasonal", family="baseline")(_CliSeasonal)


@pytest.fixture
def panel_csv(tmp_path):
    ds = pd.date_range("2024-01-01", periods=40, freq="D")
    frames = [
        pd.DataFrame(
            {"unique_id": f"s{i}", "ds": ds, "y": 50.0 + i + 0.1 * np.arange(40)}
        )
        for i in range(2)
    ]
    path = tmp_path / "panel.csv"
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


@pytest.fixture
def gtm_csv(tmp_path):
    """Messy GTM export: custom columns, duplicate dates, and a missing month.

    Two reps, monthly close dates 2024-01 .. 2025-02 with 2024-05 absent and
    two deals (100 + 25) in the final month.
    """
    months = pd.date_range("2024-01-01", periods=14, freq="MS")
    rows = []
    for rep in ("alice", "bob"):
        for m in months:
            if m == pd.Timestamp("2024-05-01"):
                continue  # rep closed nothing that month
            rows.append({"Rep": rep, "Close Date": m.strftime("%Y-%m-%d"), "Amount": 100.0})
            if m == months[-1]:
                rows.append({"Rep": rep, "Close Date": m.strftime("%Y-%m-%d"), "Amount": 25.0})
    path = tmp_path / "gtm.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def mc(monkeypatch):
    """Install a fake montecarlo module unless the real one exists."""
    try:
        import forecast_os.finance.montecarlo  # noqa: F401

        return
    except ImportError:
        pass

    mod = types.ModuleType("forecast_os.finance.montecarlo")

    class MonteCarloSimulator:
        def __init__(self, mu=0.0, sigma=0.01, seed=0):
            self.mu = mu
            self.sigma = sigma
            self.seed = seed

        @classmethod
        def from_returns(cls, returns):
            r = np.asarray(returns, dtype=float)
            return cls(mu=float(r.mean()), sigma=float(r.std()))

        def simulate(self, s0, h, n_paths=1000):
            rng = np.random.default_rng(self.seed)
            z = rng.standard_normal((n_paths, h))
            steps = np.exp((self.mu - self.sigma**2 / 2) + self.sigma * z)
            return float(s0) * np.cumprod(steps, axis=1)

        def summary(self, s0, h, n_paths=1000, levels=(5, 25, 50, 75, 95)):
            paths = self.simulate(s0, h, n_paths)
            data = {"step": np.arange(1, h + 1)}
            for lvl in levels:
                data[f"q{lvl:02d}"] = np.percentile(paths, lvl, axis=0)
            return pd.DataFrame(data)

    mod.MonteCarloSimulator = MonteCarloSimulator
    monkeypatch.setitem(sys.modules, "forecast_os.finance.montecarlo", mod)


# -- models --------------------------------------------------------------------


def test_models_lists_registered_dummy(capsys):
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "name" in out and "family" in out
    assert "_test_cli_naive" in out


def test_models_family_filter_excludes_other_families(capsys):
    assert main(["models", "--family", "financial"]) == 0
    out = capsys.readouterr().out
    assert "_test_cli_naive" not in out


# -- forecast ------------------------------------------------------------------


def test_forecast_writes_csv_h_rows_per_series(tmp_path, panel_csv):
    out_path = tmp_path / "fc.csv"
    rc = main(
        [
            "forecast", str(panel_csv), "--h", "5",
            "--model", "_test_cli_naive", "--output", str(out_path),
        ]
    )
    assert rc == 0
    out = pd.read_csv(out_path)
    assert len(out) == 2 * 5
    assert set(out["unique_id"]) == {"s0", "s1"}
    assert "_test_cli_naive" in out.columns
    assert np.isfinite(out["_test_cli_naive"]).all()


def test_forecast_prints_when_no_output(capsys, panel_csv):
    rc = main(["forecast", str(panel_csv), "--h", "3", "--model", "_test_cli_naive"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "_test_cli_naive" in out
    assert "unique_id" in out


def test_forecast_with_level_writes_interval_columns(tmp_path, panel_csv):
    out_path = tmp_path / "fc.csv"
    rc = main(
        [
            "forecast", str(panel_csv), "--h", "4", "--model", "_test_cli_naive",
            "--level", "80", "--output", str(out_path),
        ]
    )
    assert rc == 0
    out = pd.read_csv(out_path)
    assert {"_test_cli_naive-lo-80", "_test_cli_naive-hi-80"} <= set(out.columns)


def test_forecast_missing_file_returns_2(tmp_path, capsys):
    rc = main(["forecast", str(tmp_path / "nope.csv"), "--h", "3"])
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err


def test_forecast_unknown_model_returns_2(panel_csv, capsys):
    rc = main(["forecast", str(panel_csv), "--h", "3", "--model", "_test_cli_missing"])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_forecast_contract_violation_returns_2(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"unique_id": ["a"], "ds": [1]}).to_csv(bad, index=False)  # no y column
    rc = main(["forecast", str(bad), "--h", "3", "--model", "_test_cli_naive"])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


# -- panel mapping options (--id-col/--time-col/--target-col/--agg/--freq) ------


def test_forecast_gtm_csv_with_mapping_agg_and_freq(tmp_path, gtm_csv):
    out_path = tmp_path / "fc.csv"
    rc = main(
        [
            "forecast", str(gtm_csv), "--h", "3", "--model", "_test_cli_naive",
            "--id-col", "Rep", "--time-col", "Close Date", "--target-col", "Amount",
            "--agg", "sum", "--freq", "MS", "--output", str(out_path),
        ]
    )
    assert rc == 0
    out = pd.read_csv(out_path)
    assert len(out) == 2 * 3
    assert set(out["unique_id"]) == {"alice", "bob"}
    # the final month had two deals (100 + 25); naive-last forecasts their sum
    assert (out["_test_cli_naive"] == 125.0).all()
    # monthly grid continues after the last training month
    assert out["ds"].min() == "2025-03-01"


def test_forecast_freq_fills_missing_month_with_zero(tmp_path, gtm_csv):
    out_path = tmp_path / "fc.csv"
    rc = main(
        [
            "forecast", str(gtm_csv), "--h", "2", "--model", "_test_cli_mean",
            "--id-col", "Rep", "--time-col", "Close Date", "--target-col", "Amount",
            "--agg", "sum", "--freq", "MS", "--output", str(out_path),
        ]
    )
    assert rc == 0
    out = pd.read_csv(out_path)
    # 14-month grid: 12 x 100, one 125, and the missing month filled with 0
    expected = (12 * 100.0 + 125.0 + 0.0) / 14
    assert out["_test_cli_mean"].to_numpy() == pytest.approx(expected)


def test_forecast_agg_count_ignores_target_values(tmp_path, gtm_csv):
    out_path = tmp_path / "fc.csv"
    rc = main(
        [
            "forecast", str(gtm_csv), "--h", "2", "--model", "_test_cli_naive",
            "--id-col", "Rep", "--time-col", "Close Date", "--agg", "count",
            "--output", str(out_path),
        ]
    )
    assert rc == 0
    out = pd.read_csv(out_path)
    # y becomes the deal count per month; the last month has two rows
    assert (out["_test_cli_naive"] == 2.0).all()


def test_compare_gtm_csv_with_mapping(tmp_path, gtm_csv):
    out_path = tmp_path / "board.csv"
    rc = main(
        [
            "compare", str(gtm_csv), "--h", "3", "--n-windows", "2",
            "--models", "_test_cli_naive,_test_cli_mean",
            "--id-col", "Rep", "--time-col", "Close Date", "--target-col", "Amount",
            "--agg", "sum", "--freq", "MS", "-o", str(out_path),
        ]
    )
    assert rc == 0
    board = pd.read_csv(out_path)
    assert sorted(board["model"]) == ["_test_cli_mean", "_test_cli_naive"]
    assert {"mae", "rmse", "smape"} <= set(board.columns)


def test_forecast_duplicate_rows_without_agg_returns_2(gtm_csv, capsys):
    rc = main(
        [
            "forecast", str(gtm_csv), "--h", "2", "--model", "_test_cli_naive",
            "--id-col", "Rep", "--time-col", "Close Date", "--target-col", "Amount",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "duplicate" in err


def test_forecast_freq_with_duplicates_suggests_agg(gtm_csv, capsys):
    rc = main(
        [
            "forecast", str(gtm_csv), "--h", "2", "--model", "_test_cli_naive",
            "--id-col", "Rep", "--time-col", "Close Date", "--target-col", "Amount",
            "--freq", "MS",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "--agg" in err


def test_forecast_unknown_mapping_column_returns_2(panel_csv, capsys):
    rc = main(
        [
            "forecast", str(panel_csv), "--h", "3", "--model", "_test_cli_naive",
            "--id-col", "Nope",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Nope" in err


# -- forecast --param ------------------------------------------------------------


def test_forecast_param_season_length_reaches_model(tmp_path):
    pattern = [float(10 * (i + 1)) for i in range(12)]
    path = tmp_path / "seasonal.csv"
    pd.DataFrame(
        {
            "unique_id": "s0",
            "ds": pd.date_range("2022-01-01", periods=36, freq="MS"),
            "y": pattern * 3,
        }
    ).to_csv(path, index=False)
    out_path = tmp_path / "fc.csv"
    rc = main(
        [
            "forecast", str(path), "--h", "4", "--model", "_test_cli_seasonal",
            "--param", "season_length=12", "--output", str(out_path),
        ]
    )
    assert rc == 0
    out = pd.read_csv(out_path)
    # season repeats: next 4 months replay the start of the pattern; the
    # default season_length=1 would predict a constant 120 instead
    assert list(out["_test_cli_seasonal"]) == pattern[:4]


def test_forecast_param_repeatable_with_int_and_float(tmp_path, panel_csv):
    out_path = tmp_path / "fc.csv"
    rc = main(
        [
            "forecast", str(panel_csv), "--h", "3", "--model", "ridge_lag",
            "--param", "lags=6", "--param", "alpha=0.5", "--output", str(out_path),
        ]
    )
    assert rc == 0
    out = pd.read_csv(out_path)
    assert len(out) == 2 * 3
    assert np.isfinite(out["RidgeLag"]).all()


def test_forecast_param_without_equals_returns_2(panel_csv, capsys):
    rc = main(
        [
            "forecast", str(panel_csv), "--h", "3", "--model", "_test_cli_naive",
            "--param", "nonsense",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "KEY=VALUE" in err


def test_forecast_unknown_param_key_returns_2(panel_csv, capsys):
    rc = main(
        [
            "forecast", str(panel_csv), "--h", "3", "--model", "_test_cli_naive",
            "--param", "bogus_key=1",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err


# -- compare -------------------------------------------------------------------


def test_compare_prints_leaderboard(panel_csv, capsys):
    rc = main(
        [
            "compare", str(panel_csv), "--h", "4",
            "--models", "_test_cli_naive,_test_cli_mean", "--n-windows", "2",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "_test_cli_naive" in out
    assert "_test_cli_mean" in out
    assert "mae" in out


def test_compare_writes_csv(tmp_path, panel_csv):
    out_path = tmp_path / "board.csv"
    rc = main(
        [
            "compare", str(panel_csv), "--h", "4",
            "--models", "_test_cli_naive,_test_cli_mean",
            "--metrics", "mae,rmse", "--output", str(out_path),
        ]
    )
    assert rc == 0
    board = pd.read_csv(out_path)
    assert "model" in board.columns
    assert {"mae", "rmse"} <= set(board.columns)
    assert sorted(board["model"]) == ["_test_cli_mean", "_test_cli_naive"]


def test_compare_with_level_runs_end_to_end(tmp_path, panel_csv):
    out_path = tmp_path / "board.csv"
    rc = main(
        [
            "compare", str(panel_csv), "--h", "4",
            "--models", "_test_cli_naive,_test_cli_mean", "--n-windows", "2",
            "--metrics", "mae,coverage", "--level", "80",
            "--output", str(out_path),
        ]
    )
    assert rc == 0
    board = pd.read_csv(out_path)
    assert {"mae", "coverage-80"} <= set(board.columns)
    assert sorted(board["model"]) == ["_test_cli_mean", "_test_cli_naive"]
    assert board["coverage-80"].between(0.0, 1.0).all()


def test_compare_level_repeat_and_comma_list(tmp_path, panel_csv):
    out_path = tmp_path / "board.csv"
    rc = main(
        [
            "compare", str(panel_csv), "--h", "4",
            "--models", "_test_cli_naive", "--n-windows", "2",
            "--metrics", "coverage", "--level", "80,95", "--level", "60",
            "--output", str(out_path),
        ]
    )
    assert rc == 0
    board = pd.read_csv(out_path)
    assert {"coverage-60", "coverage-80", "coverage-95"} <= set(board.columns)


def test_compare_interval_metric_without_level_returns_2(panel_csv, capsys):
    rc = main(
        [
            "compare", str(panel_csv), "--h", "4",
            "--models", "_test_cli_naive", "--metrics", "coverage",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "level" in err


# -- simulate ------------------------------------------------------------------


def test_simulate_prints_quantile_summary(mc, capsys):
    rc = main(
        ["simulate", "--s0", "100", "--h", "10", "--sigma", "0.05",
         "--paths", "50", "--seed", "3"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "step" in out
    assert "q50" in out


def test_simulate_seed_reproducible(mc, capsys):
    args = ["simulate", "--s0", "100", "--h", "10", "--sigma", "0.05",
            "--paths", "50", "--seed", "3"]
    assert main(args) == 0
    first = capsys.readouterr().out
    assert main(args) == 0
    second = capsys.readouterr().out
    assert first == second
    assert main(args[:-1] + ["4"]) == 0
    third = capsys.readouterr().out
    assert first != third


def test_simulate_from_returns(mc, tmp_path, capsys):
    rets = tmp_path / "returns.csv"
    rng = np.random.default_rng(0)
    pd.DataFrame({"ds": np.arange(100), "y": rng.normal(0.001, 0.02, 100)}).to_csv(
        rets, index=False
    )
    rc = main(
        ["simulate", "--s0", "50", "--h", "5", "--from-returns", str(rets), "--seed", "1"]
    )
    assert rc == 0
    assert "q50" in capsys.readouterr().out


def test_short_output_alias_o(mc, tmp_path, panel_csv):
    """-o is a documented short alias for --output on forecast, compare, simulate."""
    fc_path = tmp_path / "fc.csv"
    rc = main(
        ["forecast", str(panel_csv), "--h", "3",
         "--model", "_test_cli_naive", "-o", str(fc_path)]
    )
    assert rc == 0
    fc = pd.read_csv(fc_path)
    assert len(fc) == 2 * 3
    assert "_test_cli_naive" in fc.columns

    board_path = tmp_path / "board.csv"
    rc = main(
        ["compare", str(panel_csv), "--h", "4",
         "--models", "_test_cli_naive,_test_cli_mean", "-o", str(board_path)]
    )
    assert rc == 0
    board = pd.read_csv(board_path)
    assert sorted(board["model"]) == ["_test_cli_mean", "_test_cli_naive"]

    sim_path = tmp_path / "sim.csv"
    rc = main(
        ["simulate", "--s0", "100", "--h", "5", "--paths", "30",
         "--seed", "1", "-o", str(sim_path)]
    )
    assert rc == 0
    assert len(pd.read_csv(sim_path)) == 5


# -- schema mappings (--mapping / mappings subcommand) -------------------------


@pytest.fixture
def hubspot_csv(tmp_path):
    """HubSpot-style deal export: one closed-won deal per month for 10 months.

    The last month has a second closed-won deal (25) on the same date, and
    February has a closed-lost deal that the recipe's filter must drop.
    """
    months = pd.date_range("2025-01-01", periods=10, freq="MS")
    rows = [
        {
            "closedate": (m + pd.Timedelta(days=4)).strftime("%Y-%m-%d"),
            "amount": 100.0,
            "dealstage": "closedwon",
            "hubspot_owner_id": "42",
        }
        for m in months
    ]
    rows.append(
        {
            "closedate": (months[-1] + pd.Timedelta(days=4)).strftime("%Y-%m-%d"),
            "amount": 25.0,
            "dealstage": "closedwon",
            "hubspot_owner_id": "7",
        }
    )
    rows.append(
        {
            "closedate": "2025-02-20",
            "amount": 999.0,
            "dealstage": "closedlost",
            "hubspot_owner_id": "42",
        }
    )
    path = tmp_path / "hubspot.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_forecast_with_mapping_hubspot_deals(tmp_path, hubspot_csv):
    out_path = tmp_path / "fc.csv"
    rc = main(
        [
            "forecast", str(hubspot_csv), "--h", "2", "--model", "_test_cli_naive",
            "--mapping", "hubspot_deals", "--output", str(out_path),
        ]
    )
    assert rc == 0
    out = pd.read_csv(out_path)
    assert len(out) == 2
    assert set(out["unique_id"]) == {"hubspot_deals"}
    # last training month sums both closed-won deals (100 + 25); closedlost
    # never reaches the panel
    assert (out["_test_cli_naive"] == 125.0).all()
    # monthly grid continues after the last training month
    assert out["ds"].min() == "2025-11-01"


def test_forecast_mapping_freq_override_takes_effect(tmp_path, hubspot_csv):
    out_path = tmp_path / "fc.csv"
    rc = main(
        [
            "forecast", str(hubspot_csv), "--h", "1", "--model", "_test_cli_mean",
            "--mapping", "hubspot_deals", "--freq", "D", "--output", str(out_path),
        ]
    )
    assert rc == 0
    out = pd.read_csv(out_path)
    # daily grid from the first to the last close date, zeros between deals;
    # the monthly default would predict (9 * 100 + 125) / 10 = 102.5 instead
    n_days = len(pd.date_range("2025-01-05", "2025-10-05", freq="D"))
    expected = (9 * 100.0 + 125.0) / n_days
    assert out["_test_cli_mean"].to_numpy() == pytest.approx(expected)
    assert out["ds"].min() == "2025-10-06"


def test_compare_with_mapping(tmp_path, hubspot_csv):
    out_path = tmp_path / "board.csv"
    rc = main(
        [
            "compare", str(hubspot_csv), "--h", "2", "--n-windows", "2",
            "--models", "_test_cli_naive,_test_cli_mean",
            "--mapping", "hubspot_deals", "-o", str(out_path),
        ]
    )
    assert rc == 0
    board = pd.read_csv(out_path)
    assert sorted(board["model"]) == ["_test_cli_mean", "_test_cli_naive"]
    assert {"mae", "rmse", "smape"} <= set(board.columns)


def test_mappings_subcommand_lists_recipes(capsys):
    assert main(["mappings"]) == 0
    out = capsys.readouterr().out
    assert "name" in out and "description" in out
    for name in ("salesforce_opportunities", "hubspot_deals", "stripe_invoices",
                 "posthog_events", "generic_events"):
        assert name in out


def test_mapping_conflicts_with_manual_panel_options(hubspot_csv, capsys):
    rc = main(
        [
            "forecast", str(hubspot_csv), "--h", "2", "--model", "_test_cli_naive",
            "--mapping", "hubspot_deals", "--id-col", "hubspot_owner_id",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "--mapping" in err and "--id-col" in err

    rc = main(
        [
            "compare", str(hubspot_csv), "--h", "2", "--models", "_test_cli_naive",
            "--mapping", "hubspot_deals", "--agg", "sum",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "--mapping" in err and "--agg" in err


def test_forecast_unknown_mapping_returns_2(hubspot_csv, capsys):
    rc = main(
        [
            "forecast", str(hubspot_csv), "--h", "2", "--model", "_test_cli_naive",
            "--mapping", "not_a_mapping",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "unknown mapping" in err


def test_simulate_writes_csv(mc, tmp_path):
    out_path = tmp_path / "sim.csv"
    rc = main(
        ["simulate", "--s0", "100", "--h", "6", "--paths", "40",
         "--seed", "2", "--output", str(out_path)]
    )
    assert rc == 0
    out = pd.read_csv(out_path)
    assert len(out) == 6
    assert {"step", "q05", "q25", "q50", "q75", "q95"} <= set(out.columns)
