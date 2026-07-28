"""CLI tests: drive ``forecast_os.cli.main(argv)`` with tmp CSVs.

Self-sufficient: registers local dummy forecasters under ``_test_``-prefixed
names at run time. The ``simulate`` command is tested against a minimal fake
``forecast_os.finance.montecarlo`` module when the real one is not built yet.
"""

import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest

from forecast_os.cli import _bucket_ds, _prepare_panel, _read_panel, build_parser, main
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


# -- --freq bucketing (regression: off-grid rows were silently deleted) --------
#
# --freq used to group on the EXACT timestamp and then reindex onto
# date_range(min, max, freq), so every row that did not literally land on the
# grid was dropped and --fill zero wrote 0.0 over it. The module docstring's
# own flagship example (a CRM export closed on arbitrary days, --freq MS) lost
# 96% of its revenue and forecast all zeros with exit status 0. Off-grid
# timestamps must be bucketed into the period that CONTAINS them — the same
# semantics --mapping already uses — and any residual misalignment must raise.


@pytest.fixture
def offgrid_crm_csv(tmp_path):
    """CRM export closed on arbitrary days of the month, as exports really are."""
    rows = []
    for rep, day in (("alice", 3), ("bob", 17)):
        for month in range(1, 8):
            rows.append(
                {
                    "Rep": rep,
                    "Close Date": f"2024-{month:02d}-{day:02d}",
                    "Amount": 100.0 * month,
                }
            )
            rows.append(
                {"Rep": rep, "Close Date": f"2024-{month:02d}-28", "Amount": 25.0}
            )
    path = tmp_path / "crm.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_freq_buckets_offgrid_rows_instead_of_dropping_them(offgrid_crm_csv):
    """Every input row must land in its own month; none may be deleted."""
    args = build_parser().parse_args(
        [
            "forecast", str(offgrid_crm_csv), "--h", "1",
            "--id-col", "Rep", "--time-col", "Close Date", "--target-col", "Amount",
            "--agg", "sum", "--freq", "MS",
        ]
    )
    raw = pd.read_csv(offgrid_crm_csv)
    panel = _prepare_panel(_read_panel(str(offgrid_crm_csv)), args)
    # no revenue is destroyed by regularizing (this returned 0.0 before the fix)
    assert panel["y"].sum() == pytest.approx(raw["Amount"].sum())
    # January is present: the grid starts at the containing month, not day 3
    assert panel["ds"].min() == pd.Timestamp("2024-01-01")
    alice = panel[panel["unique_id"] == "alice"]
    assert list(alice["y"]) == [125.0, 225.0, 325.0, 425.0, 525.0, 625.0, 725.0]


def test_forecast_offgrid_crm_export_end_to_end(tmp_path, offgrid_crm_csv):
    """The docstring's flagship example must forecast real revenue, not zeros."""
    out_path = tmp_path / "fc.csv"
    rc = main(
        [
            "forecast", str(offgrid_crm_csv), "--h", "3", "--model", "_test_cli_naive",
            "--id-col", "Rep", "--time-col", "Close Date", "--target-col", "Amount",
            "--agg", "sum", "--freq", "MS", "--output", str(out_path),
        ]
    )
    assert rc == 0
    out = pd.read_csv(out_path)
    # last month is 700 + 25; the old behaviour forecast 0.0 and still exited 0
    assert (out["_test_cli_naive"] == 725.0).all()
    assert out["ds"].min() == "2024-08-01"


def test_freq_without_agg_buckets_one_row_per_period(tmp_path):
    """One off-grid row per month needs no --agg; it belongs in that month."""
    path = tmp_path / "sparse.csv"
    pd.DataFrame(
        {
            "Rep": "alice",
            "Close Date": ["2024-01-09", "2024-02-23", "2024-04-11"],
            "Amount": [10.0, 20.0, 40.0],
        }
    ).to_csv(path, index=False)
    args = build_parser().parse_args(
        [
            "forecast", str(path), "--h", "1",
            "--id-col", "Rep", "--time-col", "Close Date", "--target-col", "Amount",
            "--freq", "MS",
        ]
    )
    panel = _prepare_panel(_read_panel(str(path)), args)
    assert list(panel["ds"]) == list(pd.date_range("2024-01-01", periods=4, freq="MS"))
    assert list(panel["y"]) == [10.0, 20.0, 0.0, 40.0]  # March gap filled


def test_freq_without_agg_reports_offgrid_duplicates(offgrid_crm_csv, capsys):
    """Two rows in one month cannot be regularized without an aggregation rule."""
    rc = main(
        [
            "forecast", str(offgrid_crm_csv), "--h", "2", "--model", "_test_cli_naive",
            "--id-col", "Rep", "--time-col", "Close Date", "--target-col", "Amount",
            "--freq", "MS",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "--agg" in err


def test_freq_buckets_end_anchored_periods(tmp_path):
    """End-anchored grids (ME) label the period end, keeping the month whole."""
    path = tmp_path / "me.csv"
    pd.DataFrame(
        {
            "Rep": "alice",
            "Close Date": ["2024-01-05", "2024-01-31", "2024-02-14"],
            "Amount": [1.0, 2.0, 4.0],
        }
    ).to_csv(path, index=False)
    args = build_parser().parse_args(
        [
            "forecast", str(path), "--h", "1",
            "--id-col", "Rep", "--time-col", "Close Date", "--target-col", "Amount",
            "--agg", "sum", "--freq", "ME",
        ]
    )
    panel = _prepare_panel(_read_panel(str(path)), args)
    assert list(panel["ds"]) == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]
    assert list(panel["y"]) == [3.0, 4.0]


@pytest.mark.parametrize("freq", ["D", "ME", "W-MON", "QE", "YE", "MS", "h"])
def test_freq_keeps_the_timezone_of_ds(freq):
    """Bucketing must not silently change ds's timezone.

    The bucketing fix routed every ds through the period-label helper, whose
    ``to_period`` hop warns 'will drop timezone information' and returns
    tz-naive labels: a zoned ds came out of --freq D/ME/W-*/QE/YE as
    datetime64[us], so the dtype and the printed offsets changed under the
    user without a word. --freq chooses a grid; it does not re-zone the data,
    and the label a row lands in is the one its LOCAL wall clock falls in.
    """
    ds = pd.to_datetime(["2024-01-08 09:30", "2024-02-14 16:00"]).tz_localize(
        "America/New_York"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the tz-drop UserWarning must not fire
        out = _bucket_ds(pd.DataFrame({"ds": ds}), freq)
    assert str(out["ds"].dt.tz) == "America/New_York"
    naive = _bucket_ds(pd.DataFrame({"ds": ds.tz_localize(None)}), freq)["ds"]
    assert list(out["ds"].dt.tz_localize(None)) == list(naive)


def test_freq_buckets_a_zone_whose_local_midnight_does_not_exist():
    """A DST spring-forward at midnight must not abort the run.

    Bucketing normalizes to local midnight, which does not exist in Santiago
    on 2024-09-08 (and Havana, Beirut, Sao Paulo have the same shape), so
    ``--freq D`` failed with 'error: 2024-09-08 00:00:00 is a nonexistent time
    due to daylight savings time' — no mention of ds or of a timezone. The day
    a 12:00 local timestamp belongs to is still that day; its label is the
    first instant the zone actually has.
    """
    ds = pd.to_datetime(["2024-09-08 12:00"]).tz_localize("America/Santiago")
    out = _bucket_ds(pd.DataFrame({"ds": ds}), "D")
    assert list(out["ds"]) == [pd.Timestamp("2024-09-08 01:00", tz="America/Santiago")]


# -- null ds (regression: --agg silently deleted undated rows) -----------------


@pytest.fixture
def undated_crm_csv(tmp_path):
    """CRM export where two open deals have no close date yet."""
    path = tmp_path / "undated.csv"
    pd.DataFrame(
        {
            "Rep": ["alice"] * 5,
            "Close Date": ["2024-01-05", "2024-02-05", "", "", "2024-03-05"],
            "Amount": [10.0, 20.0, 1000.0, 554.0, 10.0],
        }
    ).to_csv(path, index=False)
    return path


def test_agg_and_freq_reject_rows_with_a_null_ds(undated_crm_csv, capsys):
    """A blank close date must be reported, not quietly deleted.

    The module docstring promises 'nothing is dropped', but ``_aggregate``'s
    groupby defaults to dropna=True, so a row with no close date never
    reached the grid: this export sums to 1594.0 and the CLI printed a panel
    summing to 40.0 with exit status 0. ``--mapping``/``to_panel`` already
    raises on the same data, so raising here is also what makes the two paths
    agree; an undated row's period is unknowable, and only the user can say
    whether to drop or fill it.
    """
    rc = main(
        [
            "forecast", str(undated_crm_csv), "--h", "1", "--model", "_test_cli_naive",
            "--id-col", "Rep", "--time-col", "Close Date", "--target-col", "Amount",
            "--agg", "sum", "--freq", "MS",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "2 row(s)" in err and "null" in err


def test_agg_without_freq_rejects_rows_with_a_null_ds(undated_crm_csv, capsys):
    """--agg drops undated rows on its own too, with no --freq involved."""
    rc = main(
        [
            "forecast", str(undated_crm_csv), "--h", "1", "--model", "_test_cli_naive",
            "--id-col", "Rep", "--time-col", "Close Date", "--target-col", "Amount",
            "--agg", "sum",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "null" in err and "ds" in err


# -- null unique_id (regression: --agg/--freq silently deleted unidentified rows) --


@pytest.fixture
def unassigned_crm_csv(tmp_path):
    """CRM export where one deal has no owner in the Rep column."""
    path = tmp_path / "unassigned.csv"
    pd.DataFrame(
        {
            "Rep": ["alice", "alice", "", "bob", "bob"],
            "Close Date": [
                "2024-01-01", "2024-02-01", "2024-01-01", "2024-01-01", "2024-02-01",
            ],
            "Amount": [10.0, 20.0, 999.0, 5.0, 7.0],
        }
    ).to_csv(path, index=False)
    return path


def test_agg_rejects_rows_with_a_null_unique_id(unassigned_crm_csv, capsys):
    """A blank owner cell must be reported, not quietly deleted by --agg.

    What went wrong: ``_aggregate``'s groupby defaults to ``dropna=True``, so
    a row with a blank mapped id vanished with its y. This export sums to
    1041.0 and the CLI printed a panel summing to 42.0 with exit status 0 —
    the 999.0 row simply gone, no warning.

    Why raising is right: without ``--agg`` this very file already exits 2,
    because ``validate_panel`` rejects null keys ("every row must name the
    series it belongs to"). So ``--agg`` was silently *bypassing* a contract
    check the plain path enforces; the two paths must agree, and only the
    user can say whether to drop the row or assign it an owner.
    """
    rc = main(
        [
            "forecast", str(unassigned_crm_csv), "--h", "1", "--model", "_test_cli_naive",
            "--id-col", "Rep", "--time-col", "Close Date", "--target-col", "Amount",
            "--agg", "sum",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "1 row(s)" in err and "unique_id" in err
    assert "Traceback" not in err


def test_freq_rejects_rows_with_a_null_unique_id(unassigned_crm_csv, capsys):
    """--freq regularizes per series, so it dropped unidentified rows too."""
    rc = main(
        [
            "forecast", str(unassigned_crm_csv), "--h", "1", "--model", "_test_cli_naive",
            "--id-col", "Rep", "--time-col", "Close Date", "--target-col", "Amount",
            "--freq", "MS",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "unique_id" in err and "null" in err


def test_agg_keeps_every_identified_row(tmp_path, capsys):
    """The guard must not reject ordinary exports: full ids still aggregate."""
    path = tmp_path / "clean.csv"
    pd.DataFrame(
        {
            "Rep": ["alice", "alice", "alice", "bob", "bob"],
            "Close Date": [
                "2024-01-01", "2024-01-17", "2024-02-01", "2024-01-01", "2024-02-01",
            ],
            "Amount": [10.0, 999.0, 20.0, 5.0, 7.0],
        }
    ).to_csv(path, index=False)
    out_path = tmp_path / "out.csv"
    rc = main(
        [
            "forecast", str(path), "--h", "1", "--model", "_test_cli_naive",
            "--id-col", "Rep", "--time-col", "Close Date", "--target-col", "Amount",
            "--agg", "sum", "--freq", "MS", "--output", str(out_path),
        ]
    )
    assert rc == 0
    assert out_path.exists()


# -- --fill nan ------------------------------------------------------------------


def test_fill_nan_reports_a_cli_level_remedy(tmp_path, capsys):
    """--fill nan must fail with advice a shell user can act on.

    What went wrong: ``--fill nan`` leaves gap rows as NaN, but every CLI
    consumer routes through ``validate_panel``, which rejects NaN targets
    unless ``allow_missing=True`` — a library-only kwarg the CLI exposes
    nowhere. So the user was told to "pass allow_missing=True", which reads
    as actionable advice but is a dead end from the shell.

    Right behaviour: the CLI names the gaps it created and points at options
    that exist on the command line (--fill zero), while still failing (exit
    2) rather than forecasting an incomplete target.
    """
    path = tmp_path / "gap.csv"
    pd.DataFrame(
        {
            "Rep": ["alice"] * 3,
            "Close Date": ["2024-01-01", "2024-03-01", "2024-04-01"],
            "Amount": [10.0, 20.0, 30.0],
        }
    ).to_csv(path, index=False)
    rc = main(
        [
            "forecast", str(path), "--h", "1", "--model", "_test_cli_naive",
            "--id-col", "Rep", "--time-col", "Close Date", "--target-col", "Amount",
            "--agg", "sum", "--freq", "MS", "--fill", "nan",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "--fill nan" in err and "--fill zero" in err
    assert "allow_missing" not in err  # the library-only dead end is gone
    assert "Traceback" not in err
    # "reports the gaps it found" has to mean the periods, not just a count:
    # a count sends the user back to the spreadsheet to find them by hand
    assert "2024-02-01" in err and "alice" in err


def test_fill_nan_is_still_accepted_when_there_are_no_gaps(tmp_path):
    """--fill nan on a gapless panel is a no-op and must keep working."""
    path = tmp_path / "nogap.csv"
    pd.DataFrame(
        {
            "Rep": ["alice"] * 4,
            "Close Date": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
            "Amount": [10.0, 20.0, 30.0, 40.0],
        }
    ).to_csv(path, index=False)
    rc = main(
        [
            "forecast", str(path), "--h", "1", "--model", "_test_cli_naive",
            "--id-col", "Rep", "--time-col", "Close Date", "--target-col", "Amount",
            "--agg", "sum", "--freq", "MS", "--fill", "nan",
        ]
    )
    assert rc == 0


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


@pytest.mark.parametrize("metrics", ["", ",", " , "])
def test_compare_empty_metrics_returns_2_not_a_traceback(panel_csv, capsys, metrics):
    """An empty --metrics must obey the documented "error: ... exit 2" contract.

    What went wrong: ``_split_csv_arg`` turned these into ``[]``, and
    ``compare(metrics=[])`` raised ``KeyError: 'metric'`` from pandas. KeyError
    is not in main()'s caught tuple, so a full traceback reached the terminal
    and the process exited 1 — contradicting cli.py's module docstring:
    "print ``error: ...`` to stderr and exit with status 2 — never a
    traceback".
    """
    rc = main(
        [
            "compare", str(panel_csv), "--h", "4",
            "--models", "_test_cli_naive", "--metrics", metrics,
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err
    assert "metric" in err


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


@pytest.fixture
def gappy_hubspot_csv(tmp_path):
    """HubSpot-style export with a five-month hole in the middle."""
    months = [*pd.date_range("2025-01-01", periods=4, freq="MS"),
              *pd.date_range("2025-10-01", periods=4, freq="MS")]
    rows = [
        {
            "closedate": (m + pd.Timedelta(days=4)).strftime("%Y-%m-%d"),
            "amount": 100.0,
            "dealstage": "closedwon",
            "hubspot_owner_id": "42",
        }
        for m in months
    ]
    path = tmp_path / "hubspot_gappy.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_mapping_honours_fill_nan_instead_of_dropping_it(gappy_hubspot_csv, capsys):
    """--fill must reach the --mapping path, not be silently discarded.

    What went wrong: ``_prepare_panel`` returned early on the ``--mapping``
    branch with ``freq`` as the only override, so ``--fill`` never reached
    the gap handling. ``--mapping`` errors on a conflicting ``--id-col``/
    ``--time-col``/``--target-col``/``--agg`` but accepted ``--fill`` and
    dropped it, so a user auditing for holes with ``--fill nan`` got an
    exit-0 forecast built on the recipe's invented zeros — the one outcome
    the flag exists to prevent.
    """
    rc = main(
        [
            "forecast", str(gappy_hubspot_csv), "--h", "2", "--model", "_test_cli_naive",
            "--mapping", "hubspot_deals", "--fill", "nan",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "--fill nan" in err and "--fill zero" in err
    assert "2025-05-01" in err  # the first missing period, named
    assert "Traceback" not in err


def test_mapping_with_fill_nan_and_no_gaps_still_forecasts(hubspot_csv, tmp_path):
    """--fill nan is a no-op on a gapless recipe panel, exactly as on --freq."""
    out_path = tmp_path / "fc.csv"
    rc = main(
        [
            "forecast", str(hubspot_csv), "--h", "2", "--model", "_test_cli_naive",
            "--mapping", "hubspot_deals", "--fill", "nan", "--output", str(out_path),
        ]
    )
    assert rc == 0
    assert len(pd.read_csv(out_path)) == 2


def test_mapping_default_fill_still_zero_fills(gappy_hubspot_csv, tmp_path):
    """Backward compatibility: without --fill the recipe keeps zero-filling."""
    out_path = tmp_path / "fc.csv"
    rc = main(
        [
            "forecast", str(gappy_hubspot_csv), "--h", "2", "--model", "_test_cli_naive",
            "--mapping", "hubspot_deals", "--output", str(out_path),
        ]
    )
    assert rc == 0
    assert len(pd.read_csv(out_path)) == 2


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
