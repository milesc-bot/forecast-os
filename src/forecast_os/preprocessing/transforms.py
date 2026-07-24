"""Panel-to-panel transforms with per-series statistics.

All transforms are sklearn-style: ``fit(df) -> self``, ``transform(df)``,
``fit_transform(df)``, ``inverse_transform(df)``. ``inverse_transform`` also
handles forecast frames: it applies the inverse to every numeric value column
present — anything other than ``unique_id``, ``ds``, and ``cutoff``, which
covers ``y``, ``yhat``, renamed model point columns (e.g. cross-validation's
``"SES"``), and interval columns — leaving identifier and non-numeric
columns untouched.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from ..core.exceptions import ForecastOSError, NotFittedError
from ..core.types import ID_COL, TARGET_COL, TIME_COL, validate_panel

_LO_HI_RE = re.compile(r"^(lo|hi)-|-(lo|hi)-")


def _value_columns(df: pd.DataFrame) -> list[str]:
    """Columns the forward transform applies to: y, yhat, interval columns."""
    return [
        c for c in df.columns if c in (TARGET_COL, "yhat") or _LO_HI_RE.search(str(c))
    ]


def _inverse_value_columns(df: pd.DataFrame) -> list[str]:
    """Columns the inverse applies to: every numeric non-identifier column.

    Forecast and cross-validation frames rename the point column after the
    model (e.g. ``"SES"``), so the inverse cannot rely on fixed names; it
    inverts every numeric column except ``unique_id``, ``ds`` and ``cutoff``,
    leaving non-numeric passthrough columns untouched.
    """
    return [
        c
        for c in df.columns
        if c not in (ID_COL, TIME_COL, "cutoff")
        and pd.api.types.is_numeric_dtype(df[c])
    ]


class BaseTransform:
    """Common plumbing for panel transforms."""

    def fit(self, df: pd.DataFrame) -> BaseTransform:
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identity by default; stateful transforms override."""
        return df.copy()

    def _check_is_fitted(self) -> None:
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError(f"{type(self).__name__} is not fitted; call fit() first")

    def _series_params(self, store: dict[Any, Any], uid: Any) -> Any:
        try:
            return store[uid]
        except KeyError:
            raise ForecastOSError(
                f"{type(self).__name__} was not fitted on series {uid!r}"
            ) from None

    def _apply_per_series(self, df: pd.DataFrame, func, inverse: bool = False) -> pd.DataFrame:
        """Apply ``func(uid, values) -> values`` to every value column, per series."""
        self._check_is_fitted()
        if ID_COL not in df.columns:
            raise ForecastOSError(f"frame must contain the {ID_COL!r} column")
        out = df.copy()
        cols = _inverse_value_columns(out) if inverse else _value_columns(out)
        for uid, idx in out.groupby(ID_COL, sort=False).indices.items():
            for c in cols:
                vals = out[c].to_numpy(dtype=float)[idx]
                out.iloc[idx, out.columns.get_loc(c)] = func(uid, vals)
        return out


class Imputer(BaseTransform):
    """Fill missing ``y`` values per series.

    Methods: ``"interpolate"`` (linear, edges ffill/bfill), ``"ffill"``
    (forward fill, leading NaN backfilled), ``"mean"`` (series mean).
    Stateless: imputation uses the frame being transformed. No inverse.
    """

    _METHODS = ("interpolate", "ffill", "mean")

    def __init__(self, method: str = "interpolate"):
        if method not in self._METHODS:
            raise ValueError(f"unknown imputation method {method!r}; choose from {self._METHODS}")
        self.method = method

    def fit(self, df: pd.DataFrame) -> Imputer:
        validate_panel(df, allow_missing=True)
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = validate_panel(df, allow_missing=True)

        def _fill(s: pd.Series) -> pd.Series:
            if self.method == "interpolate":
                return s.interpolate(method="linear").ffill().bfill()
            if self.method == "ffill":
                return s.ffill().bfill()
            return s.fillna(s.mean())

        out[TARGET_COL] = out.groupby(ID_COL, sort=False)[TARGET_COL].transform(_fill)
        if out[TARGET_COL].isna().any():
            raise ForecastOSError(
                "imputation left NaN values (a series may be entirely missing)"
            )
        return out


class StandardScaler(BaseTransform):
    """Per-series standardization ``(y - mu) / sigma`` (sigma floored at 1e-12)."""

    def __init__(self):
        pass

    def fit(self, df: pd.DataFrame) -> StandardScaler:
        df = validate_panel(df)
        self.stats_: dict[Any, tuple[float, float]] = {}
        for uid, g in df.groupby(ID_COL, sort=True):
            y = g[TARGET_COL].to_numpy(dtype=float)
            self.stats_[uid] = (float(y.mean()), max(float(y.std()), 1e-12))
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        out = validate_panel(df)

        def _scale(uid: Any, vals: np.ndarray) -> np.ndarray:
            mu, sd = self._series_params(self.stats_, uid)
            return (vals - mu) / sd

        return self._apply_per_series(out, _scale)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        def _unscale(uid: Any, vals: np.ndarray) -> np.ndarray:
            mu, sd = self._series_params(self.stats_, uid)
            return vals * sd + mu

        return self._apply_per_series(df, _unscale, inverse=True)


class LogTransform(BaseTransform):
    """Per-series ``log(y + offset)``.

    ``offset="auto"`` picks 0 when the series minimum is positive, else
    ``1 - min(y)``; a numeric offset is used as-is for every series.
    Inverse is ``exp(x) - offset``.
    """

    def __init__(self, offset: str | float = "auto"):
        if offset != "auto" and not isinstance(offset, (int, float)):
            raise ValueError(f"offset must be 'auto' or a number, got {offset!r}")
        self.offset = offset

    def fit(self, df: pd.DataFrame) -> LogTransform:
        df = validate_panel(df)
        self.offset_: dict[Any, float] = {}
        for uid, g in df.groupby(ID_COL, sort=True):
            ymin = float(g[TARGET_COL].min())
            if self.offset == "auto":
                off = 0.0 if ymin > 0 else 1.0 - ymin
            else:
                off = float(self.offset)
                if ymin + off <= 0:
                    raise ForecastOSError(
                        f"log transform of series {uid!r} needs y + offset > 0; "
                        f"min(y) = {ymin}, offset = {off}"
                    )
            self.offset_[uid] = off
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        out = validate_panel(df)

        def _log(uid: Any, vals: np.ndarray) -> np.ndarray:
            off = self._series_params(self.offset_, uid)
            shifted = vals + off
            if (shifted <= 0).any():
                raise ForecastOSError(
                    f"log transform of series {uid!r} hit non-positive y + offset"
                )
            return np.log(shifted)

        return self._apply_per_series(out, _log)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        def _exp(uid: Any, vals: np.ndarray) -> np.ndarray:
            return np.exp(vals) - self._series_params(self.offset_, uid)

        return self._apply_per_series(df, _exp, inverse=True)


class Differencer(BaseTransform):
    """Per-series ``d``-times differencing, dropping the first ``d`` rows.

    ``inverse_transform`` integrates using stored values and supports exactly
    two kinds of frames: the **full** transformed training panel itself and
    frames whose ``ds`` all lie **after** the training series (forecast
    continuations). Arbitrary in-range slices are not invertible and raise
    :class:`ForecastOSError`.
    """

    def __init__(self, d: int = 1):
        if not isinstance(d, int) or d < 1:
            raise ValueError(f"d must be a positive integer, got {d!r}")
        self.d = d

    def fit(self, df: pd.DataFrame) -> Differencer:
        df = validate_panel(df)
        self.initials_: dict[Any, list[float]] = {}
        self.lasts_: dict[Any, list[float]] = {}
        self.last_ds_: dict[Any, Any] = {}
        self.n_train_: dict[Any, int] = {}
        self.first_diff_ds_: dict[Any, Any] = {}
        for uid, g in df.groupby(ID_COL, sort=True):
            y = g[TARGET_COL].to_numpy(dtype=float)
            if len(y) <= self.d:
                raise ForecastOSError(
                    f"Differencer(d={self.d}) needs more than {self.d} observations; "
                    f"series {uid!r} has {len(y)}"
                )
            initials, lasts = [], []
            u = y
            for i in range(self.d):
                initials.append(float(u[self.d - 1 - i]))
                lasts.append(float(u[-1]))
                u = np.diff(u)
            self.initials_[uid] = initials
            self.lasts_[uid] = lasts
            self.last_ds_[uid] = g[TIME_COL].iloc[-1]
            self.n_train_[uid] = len(y)
            self.first_diff_ds_[uid] = g[TIME_COL].iloc[self.d]
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        out = validate_panel(df)
        frames = []
        for _, g in out.groupby(ID_COL, sort=True):
            y = g[TARGET_COL].to_numpy(dtype=float)
            if len(y) <= self.d:
                raise ForecastOSError(
                    f"Differencer(d={self.d}) needs more than {self.d} observations"
                )
            tail = g.iloc[self.d :].copy()
            tail[TARGET_COL] = np.diff(y, n=self.d)
            frames.append(tail)
        return pd.concat(frames, ignore_index=True)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        def _integrate(uid: Any, vals: np.ndarray, continuation: bool) -> np.ndarray:
            store = self.lasts_ if continuation else self.initials_
            starts = self._series_params(store, uid)
            x = vals
            for i in reversed(range(self.d)):
                x = starts[i] + np.cumsum(x)
            return x

        self._check_is_fitted()
        if ID_COL not in df.columns:
            raise ForecastOSError(f"frame must contain the {ID_COL!r} column")
        out = df.copy()
        cols = _inverse_value_columns(out)
        for uid, idx in out.groupby(ID_COL, sort=False).indices.items():
            last_ds = self._series_params(self.last_ds_, uid)
            ds = out[TIME_COL].iloc[idx]
            continuation = bool((ds > last_ds).all())
            if not continuation:
                n_expected = self._series_params(self.n_train_, uid) - self.d
                first_ds = self._series_params(self.first_diff_ds_, uid)
                if len(ds) != n_expected or bool(ds.iloc[0] != first_ds):
                    raise ForecastOSError(
                        f"Differencer.inverse_transform of series {uid!r} needs "
                        f"either a forecast continuation (every ds after the "
                        f"training range) or the full transformed training panel; "
                        f"got an arbitrary in-range slice, which is not invertible"
                    )
            for c in cols:
                vals = out[c].to_numpy(dtype=float)[idx]
                out.iloc[idx, out.columns.get_loc(c)] = _integrate(uid, vals, continuation)
        return out


class LogitTransform(BaseTransform):
    """Logit transform for series bounded in fixed ``[lower, upper]``.

    Bounds come from the constructor — they are domain knowledge (win rates
    in [0, 1], utilization in [0, 100]), never fitted from data. Each of
    ``lower``/``upper`` is a number applied to every series or a
    ``{unique_id: bound}`` dict for per-series bounds. ``fit`` validates that
    every observation lies within its series' bounds and raises
    :class:`ForecastOSError` otherwise.

    The forward transform clips values into
    ``[lo + eps*(hi-lo), hi - eps*(hi-lo)]`` (so boundary values stay
    finite) and maps ``y -> log((y-lo)/(hi-y))``. The inverse is the scaled
    sigmoid ``lo + (hi-lo)*expit(x)`` with the sigmoid output clamped into
    the open unit interval (``expit`` saturates to exactly 1.0 for inputs
    above ~36.7), applied to every numeric value column of forecast frames —
    point and interval columns alike land strictly inside ``(lo, hi)``.
    """

    def __init__(
        self,
        lower: float | dict[Any, float] = 0.0,
        upper: float | dict[Any, float] = 1.0,
        eps: float = 1e-6,
    ):
        if not isinstance(lower, dict) and not isinstance(upper, dict) and not lower < upper:
            raise ValueError(f"need lower < upper, got lower={lower!r}, upper={upper!r}")
        if not 0 < eps < 0.5:
            raise ValueError(f"eps must be in (0, 0.5), got {eps!r}")
        self.lower = lower
        self.upper = upper
        self.eps = eps

    def _bounds(self, uid: Any) -> tuple[float, float]:
        def pick(bound: float | dict[Any, float], side: str) -> float:
            if isinstance(bound, dict):
                try:
                    return float(bound[uid])
                except KeyError:
                    raise ForecastOSError(
                        f"LogitTransform has no {side} bound for series {uid!r}"
                    ) from None
            return float(bound)

        lo, hi = pick(self.lower, "lower"), pick(self.upper, "upper")
        if not lo < hi:
            raise ForecastOSError(
                f"series {uid!r} bounds must satisfy lower < upper; got ({lo}, {hi})"
            )
        return lo, hi

    def fit(self, df: pd.DataFrame) -> LogitTransform:
        df = validate_panel(df)
        for uid, g in df.groupby(ID_COL, sort=True):
            lo, hi = self._bounds(uid)
            ymin, ymax = float(g[TARGET_COL].min()), float(g[TARGET_COL].max())
            if ymin < lo or ymax > hi:
                raise ForecastOSError(
                    f"series {uid!r} has values outside the logit bounds "
                    f"[{lo}, {hi}]: observed range [{ymin}, {ymax}]"
                )
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        out = validate_panel(df)

        def _logit(uid: Any, vals: np.ndarray) -> np.ndarray:
            lo, hi = self._bounds(uid)
            span = hi - lo
            v = np.clip(vals, lo + self.eps * span, hi - self.eps * span)
            return np.log((v - lo) / (hi - v))

        return self._apply_per_series(out, _logit)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        def _sigmoid(uid: Any, vals: np.ndarray) -> np.ndarray:
            lo, hi = self._bounds(uid)
            # expit saturates to exactly 1.0 for x > ~36.7 (and 0.0 for very
            # negative x), which would land the inverse ON a bound; clamp
            # into the open unit interval so results stay strictly inside.
            finfo = np.finfo(float)
            p = np.clip(expit(vals), finfo.tiny, 1.0 - finfo.epsneg)
            return lo + (hi - lo) * p

        return self._apply_per_series(df, _sigmoid, inverse=True)


def fill_gaps(df: pd.DataFrame, freq: str, fill_value: float = 0.0) -> pd.DataFrame:
    """Reindex every series onto its full regular ``freq`` grid.

    Each series is reindexed to ``date_range(min(ds), max(ds), freq=freq)``
    — per-series ranges, no cross-series padding. Inserted rows get
    ``fill_value`` in ``y`` (``np.nan`` is allowed, e.g. to impute
    afterwards) and NaN in every extra column. Irregular exports with
    missing periods silently misalign positional models (lag features,
    seasonal indexing); filling the gaps restores ds alignment.

    Raises :class:`ForecastOSError` when ``ds`` is not datetime or when an
    existing timestamp does not lie on the ``freq`` grid. Returns a valid
    sorted panel.
    """
    out = validate_panel(df, allow_missing=True)
    if not pd.api.types.is_datetime64_any_dtype(out[TIME_COL]):
        raise ForecastOSError(
            f"fill_gaps requires a datetime {TIME_COL!r} column; "
            f"got dtype {out[TIME_COL].dtype}"
        )
    columns = list(out.columns)
    frames = []
    for uid, g in out.groupby(ID_COL, sort=True):
        g = g.set_index(TIME_COL)
        grid = pd.date_range(g.index.min(), g.index.max(), freq=freq)
        off_grid = g.index.difference(grid)
        if len(off_grid) > 0:
            raise ForecastOSError(
                f"series {uid!r} has ds values off the {freq!r} grid, "
                f"e.g. {off_grid[0]}; fill_gaps cannot align them"
            )
        r = g.reindex(grid)
        r.loc[np.asarray(~grid.isin(g.index)), TARGET_COL] = fill_value
        r[ID_COL] = uid
        r.index.name = TIME_COL
        frames.append(r.reset_index()[columns])
    return validate_panel(pd.concat(frames, ignore_index=True), allow_missing=True)
