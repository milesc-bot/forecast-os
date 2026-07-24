"""The ``forecast-os`` command-line interface.

CSV in, CSV (or printed table) out::

    forecast-os models [--family FAMILY]
    forecast-os forecast data.csv --h 12 --model auto_ets --level 80
    forecast-os compare data.csv --h 12 --models naive,theta,auto_ets
    forecast-os compare data.csv --h 12 --metrics mae,coverage --level 80,95
    forecast-os simulate --s0 100 --h 30 --mu 0.0005 --sigma 0.02

Errors (bad files, unknown models, contract violations) print ``error: ...``
to stderr and exit with status 2 — never a traceback.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .core.exceptions import ForecastOSError
from .core.registry import list_models
from .core.types import TARGET_COL, TIME_COL
from .engine import ForecastEngine

__all__ = ["main", "build_parser"]


def _read_panel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TIME_COL in df.columns and not pd.api.types.is_numeric_dtype(df[TIME_COL]):
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df


def _emit(frame: pd.DataFrame, output: str | None, index: bool = False) -> None:
    if output:
        frame.to_csv(output, index=index)
    else:
        print(frame.to_string(index=index))


def _split_csv_arg(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


# -- subcommand handlers -------------------------------------------------------


def _cmd_models(args: argparse.Namespace) -> None:
    print(list_models(family=args.family).to_string(index=False))


def _cmd_forecast(args: argparse.Namespace) -> None:
    df = _read_panel(args.input)
    level = [args.level] if args.level is not None else None
    out = ForecastEngine().forecast(df, h=args.h, models=[args.model], level=level)
    _emit(out, args.output, index=False)


def _cmd_compare(args: argparse.Namespace) -> None:
    df = _read_panel(args.input)
    models = _split_csv_arg(args.models) if args.models else None
    metrics = _split_csv_arg(args.metrics)
    level = None
    if args.level:
        level = [int(part) for value in args.level for part in _split_csv_arg(value)]
    board = ForecastEngine().compare(
        df, h=args.h, n_windows=args.n_windows, metrics=metrics, models=models, level=level
    )
    _emit(board, args.output, index=True)


def _cmd_simulate(args: argparse.Namespace) -> None:
    from .finance.montecarlo import MonteCarloSimulator

    if args.from_returns:
        rets = pd.read_csv(args.from_returns)
        if TARGET_COL in rets.columns:
            returns = pd.to_numeric(rets[TARGET_COL]).to_numpy(dtype=float)
        else:
            numeric = rets.select_dtypes(include="number")
            if numeric.shape[1] == 0:
                raise ValueError(f"no numeric returns column found in {args.from_returns}")
            returns = numeric.iloc[:, -1].to_numpy(dtype=float)
        calibrated = MonteCarloSimulator.from_returns(returns)
        sim = MonteCarloSimulator(mu=calibrated.mu, sigma=calibrated.sigma, seed=args.seed)
    else:
        sim = MonteCarloSimulator(mu=args.mu, sigma=args.sigma, seed=args.seed)
    summary = sim.summary(args.s0, args.h, n_paths=args.paths)
    _emit(summary, args.output, index=False)


# -- parser --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forecast-os",
        description="Forecast OS: statistical, ML, and financial forecasting from the shell.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_models = sub.add_parser("models", help="list registered models")
    p_models.add_argument("--family", default=None, help="filter by model family")
    p_models.set_defaults(func=_cmd_models)

    p_fc = sub.add_parser("forecast", help="forecast a CSV panel (unique_id, ds, y)")
    p_fc.add_argument("input", help="input CSV path")
    p_fc.add_argument("--h", type=int, required=True, help="forecast horizon")
    p_fc.add_argument("--model", default="auto_ets", help="registered model name")
    p_fc.add_argument("--level", type=int, default=None, help="confidence level (e.g. 80)")
    p_fc.add_argument(
        "-o", "--output", default=None, help="output CSV path (default: print)"
    )
    p_fc.set_defaults(func=_cmd_forecast)

    p_cmp = sub.add_parser("compare", help="cross-validate models and print a leaderboard")
    p_cmp.add_argument("input", help="input CSV path")
    p_cmp.add_argument("--h", type=int, required=True, help="forecast horizon")
    p_cmp.add_argument("--models", default=None, help="comma-separated model names")
    p_cmp.add_argument("--n-windows", type=int, default=3, help="CV windows (default 3)")
    p_cmp.add_argument(
        "--metrics", default="mae,rmse,smape", help="comma-separated metric names"
    )
    p_cmp.add_argument(
        "--level", action="append", default=None,
        help="confidence level(s) for interval metrics such as coverage/winkler; "
        "repeat the flag or comma-separate (e.g. --level 80,95)",
    )
    p_cmp.add_argument(
        "-o", "--output", default=None, help="output CSV path (default: print)"
    )
    p_cmp.set_defaults(func=_cmd_compare)

    p_sim = sub.add_parser("simulate", help="Monte Carlo (GBM) price simulation summary")
    p_sim.add_argument("--s0", type=float, required=True, help="starting price")
    p_sim.add_argument("--h", type=int, required=True, help="simulation horizon")
    p_sim.add_argument("--mu", type=float, default=0.0, help="per-period drift")
    p_sim.add_argument("--sigma", type=float, default=0.01, help="per-period volatility")
    p_sim.add_argument("--paths", type=int, default=1000, help="number of paths")
    p_sim.add_argument("--seed", type=int, default=0, help="random seed")
    p_sim.add_argument(
        "--from-returns", default=None,
        help="CSV of historical returns to calibrate mu/sigma (overrides --mu/--sigma)",
    )
    p_sim.add_argument(
        "-o", "--output", default=None, help="output CSV path (default: print)"
    )
    p_sim.set_defaults(func=_cmd_simulate)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        import forecast_os  # noqa: F401  (importing the package registers built-in models)

        args.func(args)
    except (ForecastOSError, ValueError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
