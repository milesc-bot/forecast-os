"""Per-row currency normalization for deal/opportunity frames.

CRM deal exports arrive one row per opportunity with an ``amount`` and a
``currency`` (EUR, GBP, USD, ...). :func:`to_panel` sums those amounts blindly
— it does NOT inspect currency — so a frame mixing EUR and USD rows aggregates
into a meaningless total. That silent mixed-currency landmine is what this
module defuses:

- :func:`convert_currency` rewrites each row's ``amount`` into a single
  reporting currency using either static rates (a ``{currency: rate}`` or
  ``{(from, to): rate}`` dict) or a dated rate table (an as-of join that picks
  the latest rate on or before each row's date).
- :class:`CurrencyNormalizer` wraps that as a fit/transform step so it slots
  into a preprocessing pipeline, additionally relabeling the currency column.
- :func:`guard_single_currency` is a cheap assertion to drop in before any
  aggregation, turning the silent landmine into a loud
  :class:`~forecast_os.core.exceptions.DataContractError`.

These are DEAL-GRAIN helpers (one row per opportunity), standalone like
``finance.GARCH11`` — not ``(unique_id, ds, y)`` panel transforms — so they
operate on arbitrary deal frames rather than the panel contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..core.exceptions import DataContractError, ForecastOSError

__all__ = ["convert_currency", "CurrencyNormalizer", "guard_single_currency"]


def _same_currency(value: Any, to: str) -> bool:
    """True when a row's currency equals the reporting currency ``to``."""
    return isinstance(value, str) and value == to


def _normalize_dict_rates(rates: dict, to: str) -> dict[Any, float]:
    """Reduce a rates dict to a ``{from_currency: rate}`` mapping into ``to``.

    Accepts two shapes: plain ``{currency: rate}`` (implicitly ``-> to``) and
    ``{(from, to): rate}`` pairs, from which only the pairs targeting ``to``
    are kept. Mixing the two shapes is rejected — the intent is ambiguous.
    """
    if not rates:
        return {}
    is_tuple = [isinstance(k, tuple) for k in rates]
    if all(is_tuple):
        mapping: dict[Any, float] = {}
        for key, rate in rates.items():
            if len(key) != 2:
                raise ForecastOSError(
                    f"tuple rate key {key!r} must be a (from, to) pair of length 2"
                )
            frm, tgt = key
            if _same_currency(tgt, to):
                mapping[frm] = float(rate)
        return mapping
    if any(is_tuple):
        raise ForecastOSError(
            "rates dict mixes (from, to) tuple keys with plain currency keys; "
            "use one shape consistently"
        )
    return {frm: float(rate) for frm, rate in rates.items()}


def _dict_rates(currencies: np.ndarray, rates: dict, to: str) -> np.ndarray:
    """Per-row conversion factors from a static rates dict (rows in ``to`` -> 1.0)."""
    mapping = _normalize_dict_rates(rates, to)
    rate = np.ones(len(currencies), dtype=float)
    missing: set[str] = set()
    for i, cur in enumerate(currencies):
        if _same_currency(cur, to):
            rate[i] = 1.0
        elif cur in mapping:
            rate[i] = mapping[cur]
        else:
            missing.add(str(cur))
            rate[i] = np.nan
    if missing:
        raise DataContractError(
            f"no exchange rate to convert currency {sorted(missing)} to {to!r}; "
            f"add it to rates (a {{currency: rate}} dict or {{(from, to): rate}} "
            f"pairs)"
        )
    return rate


def _asof_rates(
    df: pd.DataFrame,
    currencies: np.ndarray,
    rates: pd.DataFrame,
    to: str,
    as_of_col: str,
    date_col: str,
    currency_col: str,
) -> np.ndarray:
    """Per-row conversion factors via an as-of join on a dated rate table.

    Each row is matched to the latest rate for its currency whose ``as_of_col``
    date is on or before the row's ``date_col`` date. Rows already in ``to``
    pass through at 1.0 without needing a rate row.
    """
    needed = [currency_col, as_of_col, "rate"]
    missing_cols = [c for c in needed if c not in rates.columns]
    if missing_cols:
        raise DataContractError(
            f"dated rates frame is missing column(s) {missing_cols}; it needs "
            f"[{currency_col!r}, {as_of_col!r}, 'rate']"
        )
    if date_col not in df.columns:
        raise DataContractError(
            f"missing date column {date_col!r} in df for the as-of currency join"
        )
    try:
        row_dt = pd.to_datetime(df[date_col])
    except (ValueError, TypeError) as exc:
        raise DataContractError(
            f"date column {date_col!r} contains unparseable dates: {exc}"
        ) from exc
    try:
        rate_dt = pd.to_datetime(rates[as_of_col])
    except (ValueError, TypeError) as exc:
        raise DataContractError(
            f"rates date column {as_of_col!r} contains unparseable dates: {exc}"
        ) from exc
    if row_dt.isna().any():
        raise DataContractError(
            f"date column {date_col!r} has null/unparseable date(s); the as-of "
            f"currency join needs a date on every row"
        )
    if rate_dt.isna().any():
        raise DataContractError(
            f"rates date column {as_of_col!r} has null/unparseable date(s)"
        )
    try:
        rate_val = pd.to_numeric(rates["rate"]).to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise DataContractError(f"rates 'rate' column must be numeric: {exc}") from exc

    left = pd.DataFrame(
        {
            "_cur": pd.Series(currencies, dtype=object),
            "_dt": row_dt.to_numpy(),
            "_orig": np.arange(len(df)),
        }
    ).sort_values("_dt", kind="stable")
    right = pd.DataFrame(
        {
            "_cur": pd.Series(rates[currency_col].to_numpy(), dtype=object),
            "_dt": rate_dt.to_numpy(),
            "_rate": rate_val,
        }
    ).sort_values("_dt", kind="stable")
    # Force a shared object dtype on the by-key: pandas can infer StringDtype on
    # one side, and merge_asof rejects mismatched merge-key dtypes.
    left["_cur"] = left["_cur"].astype(object)
    right["_cur"] = right["_cur"].astype(object)
    merged = pd.merge_asof(left, right, on="_dt", by="_cur", direction="backward")
    merged = merged.sort_values("_orig")
    rate = merged["_rate"].to_numpy(dtype=float)

    is_to = np.array([_same_currency(cur, to) for cur in currencies])
    rate = np.where(is_to, 1.0, rate)
    missing_mask = np.isnan(rate)
    if missing_mask.any():
        dates = row_dt.to_numpy()
        pairs = sorted(
            {
                (str(cur), str(pd.Timestamp(dt).date()))
                for cur, dt, bad in zip(currencies, dates, missing_mask)
                if bad
            }
        )
        raise DataContractError(
            f"no exchange rate on or before the row date for {pairs} to convert "
            f"to {to!r}; extend the dated rates table to cover these"
        )
    return rate


def convert_currency(
    df: pd.DataFrame,
    rates: dict | pd.DataFrame,
    amount_col: str = "amount",
    currency_col: str = "currency",
    to: str = "USD",
    as_of_col: str | None = None,
    date_col: str | None = None,
) -> pd.DataFrame:
    """Convert each row's ``amount`` into the reporting currency ``to``.

    Returns a copy of ``df`` with ``amount_col`` rewritten to values expressed
    in ``to``. A row whose currency already equals ``to`` passes through
    unchanged (rate 1.0). The currency column is left as-is — only the amount
    is rewritten; use :class:`CurrencyNormalizer` when you also want the
    currency label relabeled to ``to``.

    Two rate sources are supported:

    - **Static** ``rates`` dict: either ``{currency: rate}`` (each rate is that
      currency's value in ``to``, so ``amount_in_to = amount * rate``) or
      ``{(from, to): rate}`` pairs (only the pairs whose second element equals
      ``to`` are used). The two shapes may not be mixed.
    - **Dated** ``rates`` DataFrame with columns ``[currency_col, as_of_col,
      'rate']``: pass ``as_of_col`` (the rate's effective-date column in the
      rates frame) and ``date_col`` (each row's date column in ``df``) to run
      an as-of join — every row takes the latest rate for its currency dated on
      or before the row's date.

    Raises
    ------
    DataContractError
        If ``df`` is not a DataFrame, an expected column is missing, the amount
        column is non-numeric, or a ``(currency[, date])`` combination has no
        matching rate (the offending values are named).
    ValueError
        If the dated/static configuration is inconsistent — e.g. only one of
        ``as_of_col`` / ``date_col`` is given, a DataFrame is passed without
        those columns, or a dict is passed alongside them.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataContractError(f"expected a pandas DataFrame, got {type(df).__name__}")
    for col in (amount_col, currency_col):
        if col not in df.columns:
            raise DataContractError(
                f"missing column {col!r}; convert_currency needs an amount column "
                f"({amount_col!r}) and a currency column ({currency_col!r})"
            )

    dated = as_of_col is not None or date_col is not None
    if dated and (as_of_col is None or date_col is None):
        raise ValueError(
            "the as-of currency join needs both as_of_col (the rate-date column "
            "in the rates frame) and date_col (the row-date column in df)"
        )
    if dated and not isinstance(rates, pd.DataFrame):
        raise ValueError(
            "as_of_col/date_col were given, so rates must be a DataFrame of dated "
            f"rates [{currency_col!r}, as_of_col, 'rate'], got {type(rates).__name__}"
        )
    if isinstance(rates, pd.DataFrame) and not dated:
        raise ValueError(
            "a rates DataFrame is a dated rate table; pass as_of_col and date_col "
            "to run the as-of join, or pass a {currency: rate} dict for static rates"
        )
    if not dated and not isinstance(rates, dict):
        raise ForecastOSError(
            f"rates must be a {{currency: rate}} dict or a dated DataFrame, got "
            f"{type(rates).__name__}"
        )

    out = df.copy()
    try:
        amounts = out[amount_col].to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise DataContractError(
            f"amount column {amount_col!r} must be numeric: {exc}"
        ) from exc
    currencies = out[currency_col].to_numpy(dtype=object)

    if dated:
        rate = _asof_rates(
            out, currencies, rates, to, as_of_col, date_col, currency_col
        )
    else:
        rate = _dict_rates(currencies, rates, to)

    out[amount_col] = amounts * rate
    return out


class CurrencyNormalizer:
    """Fit/transform wrapper around :func:`convert_currency` for pipelines.

    Holds static ``rates`` (a ``{currency: rate}`` or ``{(from, to): rate}``
    dict) and a reporting currency ``to``. ``fit`` is a no-op returning self
    (the rates are supplied at construction), and ``transform`` returns the
    frame with ``amount_col`` converted into ``to`` AND ``currency_col``
    relabeled to ``to`` — so the output is a clean single-currency frame that
    :func:`guard_single_currency` accepts and :func:`to_panel` can safely
    aggregate. Constructor arguments are stored as same-named attributes.
    """

    def __init__(
        self,
        rates: dict,
        currency_col: str = "currency",
        to: str = "USD",
        amount_col: str = "amount",
    ):
        self.rates = rates
        self.currency_col = currency_col
        self.to = to
        self.amount_col = amount_col

    def fit(self, df: pd.DataFrame | None = None) -> CurrencyNormalizer:
        """No-op fit (rates are fixed at construction); returns ``self``."""
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert amounts to ``to`` and relabel the currency column to ``to``."""
        out = convert_currency(
            df,
            self.rates,
            amount_col=self.amount_col,
            currency_col=self.currency_col,
            to=self.to,
        )
        out[self.currency_col] = self.to
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def guard_single_currency(df: pd.DataFrame, currency_col: str = "currency") -> None:
    """Raise if a deal frame mixes currencies; a cheap pre-aggregation guard.

    :func:`to_panel` does NOT auto-detect currency — it sums ``amount`` across
    whatever rows it is given — so mixed-currency amounts must be normalized to
    one reporting currency first (see :func:`convert_currency` /
    :class:`CurrencyNormalizer`). Call this immediately before aggregating to
    make that silent landmine loud: it raises
    :class:`~forecast_os.core.exceptions.DataContractError` when ``currency_col``
    holds more than one distinct non-null value, and is a no-op (returns
    ``None``) for a single currency or an all-null column.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataContractError(f"expected a pandas DataFrame, got {type(df).__name__}")
    if currency_col not in df.columns:
        raise DataContractError(
            f"missing currency column {currency_col!r}; guard_single_currency "
            f"checks that a deal frame carries a single reporting currency"
        )
    distinct = pd.unique(df[currency_col].dropna())
    if len(distinct) > 1:
        raise DataContractError(
            f"frame mixes {len(distinct)} currencies {sorted(map(str, distinct))}; "
            f"normalize to one reporting currency first (convert_currency / "
            f"CurrencyNormalizer) — to_panel does NOT detect currency and would "
            f"sum mixed-currency amounts into a meaningless total"
        )
