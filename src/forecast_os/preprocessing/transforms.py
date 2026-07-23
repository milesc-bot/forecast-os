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
