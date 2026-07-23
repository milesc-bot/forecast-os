"""The universal panel data contract: long-format (unique_id, ds, y).

Every model, transform, and evaluation utility in forecast-os speaks this
language. ``unique_id`` identifies a series, ``ds`` is the time index (either
datetime-like or numeric), and ``y`` is the observed value.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .exceptions import DataContractError

ID_COL = "unique_id"
TIME_COL = "ds"
TARGET_COL = "y"
REQUIRED_COLS = (ID_COL, TIME_COL, TARGET_COL)


def validate_panel(df: pd.DataFrame, allow_missing: bool = False) -> pd.DataFrame:
    """Validate and normalize a panel DataFrame.

    Returns a copy sorted by (unique_id, ds) with ``y`` coerced to float.
    Raises :class:`DataContractError` on missing columns, unparseable types,
    duplicate (unique_id, ds) pairs, or NaN targets (unless ``allow_missing``).
    """
    if not isinstance(df, pd.DataFrame):
        raise DataContractError(f"expected a pandas DataFrame, got {type(df).__name__}")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise DataContractError(
            f"missing required column(s) {missing}; the contract is "
            f"(unique_id, ds, y) in long format"
        )
    if len(df) == 0:
        raise DataContractError("panel is empty")

    out = df.copy()
    try:
        out[TARGET_COL] = pd.to_numeric(out[TARGET_COL])
    except (ValueError, TypeError) as exc:
        raise DataContractError(f"column 'y' must be numeric: {exc}") from exc
    out[TARGET_COL] = out[TARGET_COL].astype(float)

    ds = out[TIME_COL]
    if not (
        pd.api.types.is_datetime64_any_dtype(ds)
        or pd.api.types.is_numeric_dtype(ds)
    ):
        try:
            out[TIME_COL] = pd.to_datetime(ds)
        except (ValueError, TypeError) as exc:
            raise DataContractError(
                f"column 'ds' must be datetime-like or numeric: {exc}"
            ) from exc

    out = out.sort_values([ID_COL, TIME_COL], kind="stable").reset_index(drop=True)

    dup = out.duplicated([ID_COL, TIME_COL])
    if dup.any():
        first = out.loc[dup, [ID_COL, TIME_COL]].iloc[0]
        raise DataContractError(
            f"duplicate (unique_id, ds) pair: ({first[ID_COL]!r}, {first[TIME_COL]!r})"
        )
    if not allow_missing and out[TARGET_COL].isna().any():
        n = int(out[TARGET_COL].isna().sum())
        raise DataContractError(
            f"column 'y' contains {n} NaN value(s); impute them first "
            f"(see forecast_os.preprocessing) or pass allow_missing=True"
        )
    return out


def infer_step(ds: pd.Series) -> Any:
    """Infer the time step of a single series' ``ds`` column.

    For datetime ``ds`` returns a pandas frequency string when one can be
    inferred, otherwise the median ``Timedelta`` between observations.
    For numeric ``ds`` returns the median numeric step (1 for length-1 series).
    """
    ds = ds.reset_index(drop=True)
    if pd.api.types.is_datetime64_any_dtype(ds):
        if len(ds) >= 3:
            freq = pd.infer_freq(pd.DatetimeIndex(ds))
            if freq is not None:
                return freq
        if len(ds) >= 2:
            return pd.Series(pd.DatetimeIndex(ds)).diff().dropna().median()
        return pd.Timedelta(days=1)
    diffs = pd.Series(ds).diff().dropna()
    if len(diffs) == 0:
        return 1
    step = float(diffs.median())
    return int(step) if step == int(step) else step


def future_ds(last_ds: Any, h: int, step: Any) -> np.ndarray | pd.DatetimeIndex:
    """Generate ``h`` future time stamps after ``last_ds`` with the given step."""
    if isinstance(last_ds, (pd.Timestamp, np.datetime64)):
        return pd.date_range(start=last_ds, periods=h + 1, freq=step)[1:]
    return np.asarray(last_ds) + step * np.arange(1, h + 1)


def to_panel(
    y: np.ndarray | list[float],
    ds: Any = None,
    unique_id: str = "series-0",
) -> pd.DataFrame:
    """Build a single-series panel from a plain array (convenience helper)."""
    y = np.asarray(y, dtype=float)
    if ds is None:
        ds = np.arange(len(y))
    return pd.DataFrame({ID_COL: unique_id, TIME_COL: ds, TARGET_COL: y})
