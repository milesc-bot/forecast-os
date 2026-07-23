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


@pytest.fixture(autouse=True)
def _register_dummies():
    # re-registering the same class is a no-op, so autouse is safe
    register("_test_cli_naive", family="baseline")(_CliNaive)
    register("_test_cli_mean", family="baseline")(_CliMean)


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
