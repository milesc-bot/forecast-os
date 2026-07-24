"""The ``forecast-os`` command-line interface.

CSV in, CSV (or printed table) out::

    forecast-os models [--family FAMILY]
    forecast-os mappings
    forecast-os forecast data.csv --h 12 --model auto_ets --level 80
    forecast-os forecast crm.csv --h 6 --id-col Rep --time-col "Close Date" \\
        --target-col Amount --agg sum --freq MS
    forecast-os forecast deals.csv --h 6 --mapping hubspot_deals
    forecast-os forecast data.csv --h 12 --model ridge_lag --param lags=12
    forecast-os compare data.csv --h 12 --models naive,theta,auto_ets
    forecast-os compare data.csv --h 12 --metrics mae,coverage --level 80,95
    forecast-os simulate --s0 100 --h 30 --mu 0.0005 --sigma 0.02

``forecast`` and ``compare`` share panel-mapping options for messy exports:
``--id-col/--time-col/--target-col`` rename input columns to the
``(unique_id, ds, y)`` contract, ``--agg {sum,mean,count}`` collapses
duplicate ``(unique_id, ds)`` rows (``count`` ignores target values), and
``--freq FREQ`` reindexes each series to a regular grid from its first to its
last timestamp, filling gaps per ``--fill {zero,nan}`` (default ``zero``).
Timestamps must land on the ``--freq`` grid (e.g. month starts for ``MS``).

Alternatively ``--mapping NAME`` applies a named platform recipe (list them
with ``forecast-os mappings``) that renames, filters, and aggregates the raw
export in one step. It replaces the manual options above — combining it with
``--id-col/--time-col/--target-col/--agg`` is an error — and ``--freq``
becomes an override of the recipe's frequency.

Errors (bad files, unknown models, contract violations) print ``error: ...``
to stderr and exit with status 2 — never a traceback.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import pandas as pd

from .core.exceptions import ForecastOSError
from .core.registry import get_model, list_models
from .core.types import ID_COL, TARGET_COL, TIME_COL
from .engine import ForecastEngine

__all__ = ["main", "build_parser"]


def _ensure_datetime_ds(df: pd.DataFrame) -> pd.DataFrame:
    if TIME_COL in df.columns and not (
        pd.api.types.is_numeric_dtype(df[TIME_COL])
        or pd.api.types.is_datetime64_any_dtype(df[TIME_COL])
    ):
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df


def _read_panel(path: str) -> pd.DataFrame:
    return _ensure_datetime_ds(pd.read_csv(path))


def _emit(frame: pd.DataFrame, output: str | None, index: bool = False) -> None:
    if output:
        frame.to_csv(output, index=index)
    else:
        print(frame.to_string(index=index))


def _split_csv_arg(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


# -- panel mapping (--id-col/--time-col/--target-col/--agg/--freq) -------------


def _aggregate(df: pd.DataFrame, how: str) -> pd.DataFrame:
    """Collapse duplicate (unique_id, ds) rows; ``count`` ignores target values."""
    for col in (ID_COL, TIME_COL):
        if col not in df.columns:
            raise ValueError(
                f"--agg requires a {col!r} column; map one with --id-col/--time-col"
            )
    if how == "count":
        counted = df.groupby([ID_COL, TIME_COL], as_index=False).size()
        return counted.rename(columns={"size": TARGET_COL})
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"--agg {how} requires a {TARGET_COL!r} column; map one with --target-col"
        )
    return df.groupby([ID_COL, TIME_COL], as_index=False)[TARGET_COL].agg(how)


def _regularize(df: pd.DataFrame, freq: str, fill: str) -> pd.DataFrame:
    """Reindex each series to a regular ``freq`` grid over its own ds range.

    Gaps get ``y = 0.0`` (``fill="zero"``, the GTM default: a period with no
    rows is a period with no bookings) or ``NaN`` (``fill="nan"``).
    """
    for col in (ID_COL, TIME_COL, TARGET_COL):
        if col not in df.columns:
            raise ValueError(
                f"--freq requires a {col!r} column; map one with "
                f"--id-col/--time-col/--target-col"
            )
    if not pd.api.types.is_datetime64_any_dtype(df[TIME_COL]):
        raise ValueError("--freq requires a datetime 'ds' column")
    frames = []
    for uid, g in df.groupby(ID_COL, sort=True):
        s = g.sort_values(TIME_COL).set_index(TIME_COL)[TARGET_COL]
        if s.index.duplicated().any():
            raise ValueError(
                f"series {uid!r} has duplicate ds values; pass --agg to collapse "
                f"them before --freq"
            )
        grid = pd.date_range(s.index.min(), s.index.max(), freq=freq)
        s = s.reindex(grid)
        if s.notna().sum() == 0:
            raise ValueError(
                f"--freq {freq!r} does not align with the ds values of series {uid!r}"
            )
        if fill == "zero":
            s = s.fillna(0.0)
        frames.append(
            pd.DataFrame({ID_COL: uid, TIME_COL: grid, TARGET_COL: s.to_numpy(dtype=float)})
        )
    return pd.concat(frames, ignore_index=True)


def _prepare_panel(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Apply the shared panel-mapping options before the engine validates."""
    if args.mapping:
        conflicts = [
            flag
            for flag, value in (
                ("--id-col", args.id_col),
                ("--time-col", args.time_col),
                ("--target-col", args.target_col),
                ("--agg", args.agg),
            )
            if value
        ]
        if conflicts:
            raise ValueError(
                f"--mapping replaces the manual panel options; drop "
                f"{', '.join(conflicts)} (--freq is the only --mapping override)"
            )
        from .connectors import mappings  # noqa: F401  (registers platform recipes)
        from .connectors.base import apply_mapping

        overrides = {"freq": args.freq} if args.freq else {}
        return apply_mapping(df, args.mapping, **overrides)
    rename = {}
    for source, target in (
        (args.id_col, ID_COL),
        (args.time_col, TIME_COL),
        (args.target_col, TARGET_COL),
    ):
        if source and source != target:
            if source not in df.columns:
                raise ValueError(
                    f"column {source!r} not found in input; available: {list(df.columns)}"
                )
            rename[source] = target
    if rename:
        clobbered = [c for c in rename.values() if c in df.columns]
        df = df.drop(columns=clobbered).rename(columns=rename)
    df = _ensure_datetime_ds(df)
    if args.agg:
        df = _aggregate(df, args.agg)
    if args.freq:
        df = _regularize(df, args.freq, args.fill)
    return df


def _coerce_value(text: str) -> Any:
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def _parse_params(pairs: list[str] | None) -> dict[str, Any]:
    """Parse repeated ``--param KEY=VALUE`` flags; values become int/float/str."""
    params: dict[str, Any] = {}
    for pair in pairs or ():
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"invalid --param {pair!r}; expected KEY=VALUE (e.g. lags=12)")
        params[key.strip()] = _coerce_value(value.strip())
    return params


# -- subcommand handlers -------------------------------------------------------


def _cmd_models(args: argparse.Namespace) -> None:
    print(list_models(family=args.family).to_string(index=False))


def _cmd_mappings(args: argparse.Namespace) -> None:
    from .connectors import mappings  # noqa: F401  (registers platform recipes)
    from .connectors.base import list_mappings

    print(list_mappings().to_string(index=False))


def _cmd_forecast(args: argparse.Namespace) -> None:
    df = _prepare_panel(_read_panel(args.input), args)
    level = [args.level] if args.level is not None else None
    params = _parse_params(args.param)
    try:
        model = get_model(args.model, **params)
    except TypeError as exc:
        raise ValueError(f"invalid --param for model {args.model!r}: {exc}") from exc
    out = ForecastEngine().forecast(df, h=args.h, models=[model], level=level)
    _emit(out, args.output, index=False)


def _cmd_compare(args: argparse.Namespace) -> None:
    df = _prepare_panel(_read_panel(args.input), args)
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


def _add_panel_options(p: argparse.ArgumentParser) -> None:
    """Panel-mapping options shared by ``forecast`` and ``compare``."""
    p.add_argument(
        "--mapping", default=None, metavar="NAME",
        help="named platform recipe (see `forecast-os mappings`) that renames, filters, "
        "and aggregates the raw export; replaces --id-col/--time-col/--target-col/--agg "
        "(--freq still overrides the recipe's frequency)",
    )
    p.add_argument("--id-col", default=None, help="input column to use as unique_id")
    p.add_argument("--time-col", default=None, help="input column to use as ds")
    p.add_argument("--target-col", default=None, help="input column to use as y")
    p.add_argument(
        "--agg", choices=("sum", "mean", "count"), default=None,
        help="aggregate duplicate (unique_id, ds) rows; count ignores target values",
    )
    p.add_argument(
        "--freq", default=None,
        help="pandas frequency (e.g. MS, W-MON) to reindex each series to a regular grid",
    )
    p.add_argument(
        "--fill", choices=("zero", "nan"), default="zero",
        help="fill for gaps created by --freq (default: zero)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forecast-os",
        description="Forecast OS: statistical, ML, and financial forecasting from the shell.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_models = sub.add_parser("models", help="list registered models")
    p_models.add_argument("--family", default=None, help="filter by model family")
    p_models.set_defaults(func=_cmd_models)

    p_map = sub.add_parser("mappings", help="list registered platform schema mappings")
    p_map.set_defaults(func=_cmd_mappings)

    p_fc = sub.add_parser("forecast", help="forecast a CSV panel (unique_id, ds, y)")
    p_fc.add_argument("input", help="input CSV path")
    p_fc.add_argument("--h", type=int, required=True, help="forecast horizon")
    p_fc.add_argument("--model", default="auto_ets", help="registered model name")
    p_fc.add_argument(
        "--param", action="append", default=None, metavar="KEY=VALUE",
        help="constructor parameter for --model (repeatable); "
        "values are parsed as int/float/str",
    )
    p_fc.add_argument("--level", type=int, default=None, help="confidence level (e.g. 80)")
    _add_panel_options(p_fc)
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
    _add_panel_options(p_cmp)
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
