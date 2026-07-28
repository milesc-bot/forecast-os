"""Calendar and Fourier feature engineering on the panel contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.exceptions import ForecastOSError
from ..core.types import ID_COL, TIME_COL

#: Supported calendar features and their cycle length for sin/cos encoding.
_PERIODS = {
    "dayofweek": 7.0,
    "day": 31.0,
    "month": 12.0,
    "quarter": 4.0,
    "dayofyear": 365.25,
    "hour": 24.0,
    "minute": 60.0,
}


def calendar_features(
    df: pd.DataFrame,
    features: tuple[str, ...] = ("dayofweek", "month", "dayofyear"),
    cyclical: bool = True,
) -> pd.DataFrame:
    """Append calendar features derived from a datetime ``ds`` column.

    With ``cyclical=True`` each feature becomes a ``{f}_sin``/``{f}_cos``
    pair (period-aware encoding); otherwise the raw integer column ``{f}``
    is appended. Raises :class:`ForecastOSError` when ``ds`` is not datetime,
    and — on the ``cyclical=False`` branch only — when it carries a ``NaT``.
    The cyclical encoding propagates ``NaT`` as ``NaN`` sin/cos values.
    """
    if TIME_COL not in df.columns:
        raise ForecastOSError(f"frame must contain the {TIME_COL!r} column")
    if not pd.api.types.is_datetime64_any_dtype(df[TIME_COL]):
        raise ForecastOSError(
            f"calendar_features requires a datetime {TIME_COL!r} column; "
            f"got dtype {df[TIME_COL].dtype}"
        )
    # A missing timestamp has no month and no weekday. Left alone the integer
    # branch casts NaN to 0, fabricating a month that does not exist and a
    # Monday indistinguishable from a real one, so refuse it there. The
    # cyclical branch fabricates nothing — NaN flows through sin/cos and stays
    # NaN — so it keeps accepting NaT, as it always has.
    if not cyclical:
        n_nat = int(df[TIME_COL].isna().sum())
        if n_nat:
            raise ForecastOSError(
                f"calendar_features(cyclical=False) requires a {TIME_COL!r} value "
                f"on every row (the integer encoding would turn {n_nat} NaT into "
                f"a fabricated 0). Drop or repair those rows first."
            )
    out = df.copy()
    dt = out[TIME_COL].dt
    for f in features:
        if f not in _PERIODS:
            raise ValueError(
                f"unknown calendar feature {f!r}; choose from {sorted(_PERIODS)}"
            )
        vals = getattr(dt, f).to_numpy(dtype=float)
        if cyclical:
            ang = 2 * np.pi * vals / _PERIODS[f]
            out[f"{f}_sin"] = np.sin(ang)
            out[f"{f}_cos"] = np.cos(ang)
        else:
            out[f] = vals.astype(int)
    return out


def fourier_features(df: pd.DataFrame, season_length: int, k: int = 3) -> pd.DataFrame:
    """Append Fourier seasonal terms computed on per-series integer position.

    Adds ``fourier_s{season_length}_sin{i}`` / ``..._cos{i}`` for
    ``i = 1..k``. The frame is returned sorted by (unique_id, ds).
    """
    if season_length < 2:
        raise ValueError(f"season_length must be >= 2, got {season_length}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    for col in (ID_COL, TIME_COL):
        if col not in df.columns:
            raise ForecastOSError(f"frame must contain the {col!r} column")
    out = df.sort_values([ID_COL, TIME_COL], kind="stable").reset_index(drop=True)
    t = out.groupby(ID_COL, sort=False).cumcount().to_numpy(dtype=float)
    for i in range(1, k + 1):
        ang = 2 * np.pi * i * t / season_length
        out[f"fourier_s{season_length}_sin{i}"] = np.sin(ang)
        out[f"fourier_s{season_length}_cos{i}"] = np.cos(ang)
    return out
