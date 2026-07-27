"""The connector contract: sources fetch records, mappings shape them.

A **source** knows how to fetch raw records (rows of opportunities, deals,
invoices, events) from somewhere — a file, a REST API, a warehouse query.
A **schema mapping** is a declarative recipe that turns a platform's export
shape (its column names, its status values) into arguments for
:func:`forecast_os.gtm.to_panel`. The two compose::

    HubSpotSource(token=...).to_panel()                # fetch + built-in recipe
    apply_mapping(pd.read_csv("deals.csv"), "hubspot_deals", freq="W")

Mappings are registered like models, so platform recipes are discoverable
(:func:`list_mappings`) and third-party packages can add their own.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace

import pandas as pd

from ..core.exceptions import DataContractError
from ..gtm.events import to_panel

__all__ = [
    "SchemaMapping",
    "Source",
    "apply_mapping",
    "get_mapping",
    "list_mappings",
    "register_mapping",
]


#: The accounting spelling of a negative amount: "(500)", "$(500)", "($500)".
#: The captured body is validated like any other value before it is negated.
_ACCOUNTING_NEG_RE = re.compile(r"^\s*([$€£¥]?\s*)\(\s*(.*?)\s*\)\s*$")


def _numeric_values(values: pd.Series, mapping_name: str, column: str) -> pd.Series:
    """Coerce a text-typed value column to float, or raise naming the offender.

    Only file sources clean their records: a REST payload (HubSpot returns
    every deal property as a JSON string) or a warehouse TEXT column arrives
    here as text, and aggregating text CONCATENATES it — ``"1000"`` and
    ``"2000"`` summed to 10002000.0 instead of 3000.0. Money formatting is
    accepted so all four source types share one numeric contract, using the
    CSV/Parquet cleaner's own pattern: a value is validated BEFORE its
    separators are stripped, because stripping ``,`` unconditionally turns
    the de-DE ``"1.000,50"`` into 1.0005 — a plausible answer a thousand
    times too small. Anything that pattern does not recognise raises instead.
    Blank strings (an unset property) are treated as missing, which the
    skipna aggregations then ignore.
    """
    if pd.api.types.is_numeric_dtype(values):
        return values
    # local import: connectors.files imports this module at import time
    from .files import _NUMERIC_TEXT_RE, _STRIP_RE

    text = values.astype("string").str.strip()
    text = text.where(text.str.len() > 0)  # "" is an unset value, not a number
    body = text.str.replace(_ACCOUNTING_NEG_RE, r"\1\2", regex=True)
    negative = text.str.match(_ACCOUNTING_NEG_RE).fillna(False)
    bad = values[~(body.str.match(_NUMERIC_TEXT_RE).fillna(True))]
    if len(bad) > 0:
        raise DataContractError(
            f"mapping {mapping_name!r} cannot aggregate value column {column!r}: "
            f"{bad.iloc[0]!r} is not a number (dtype {values.dtype}). Expected "
            "digits with an optional sign, currency symbol, ',' thousands "
            "separators and a '.' decimal point. Map a numeric column or clean "
            "the values before applying the mapping"
        )
    numbers = pd.to_numeric(body.str.replace(_STRIP_RE, "", regex=True))
    return numbers.mask(negative, -numbers).astype(float)


@dataclass(frozen=True)
class SchemaMapping:
    """Declarative recipe: platform export shape -> (unique_id, ds, y) panel.

    ``renames`` maps source column names to the names used by ``id_cols`` /
    ``date_col`` / ``value_col``. ``filters`` keeps only rows whose column
    value is in the allowed set (e.g. closed-won deals); a filter column
    missing after renames is an error (override ``filters={}`` to disable
    filtering for exports without that column). ``date_unit`` /
    ``date_format`` describe numeric date columns (epoch unit or strftime
    format — see :func:`~forecast_os.gtm.events.to_panel`). Any
    :func:`~forecast_os.gtm.events.to_panel` argument can be overridden at
    apply time.
    """

    name: str
    description: str
    date_col: str
    id_cols: tuple[str, ...] = ()
    value_col: str | None = None
    renames: dict[str, str] = field(default_factory=dict)
    filters: dict[str, tuple] = field(default_factory=dict)
    freq: str = "MS"
    agg: str = "sum"
    fill_value: float = 0.0
    span: str = "series"
    sep: str = "/"
    date_unit: str | None = None
    date_format: str | None = None

    def apply(self, records: pd.DataFrame, **overrides) -> pd.DataFrame:
        """Rename, filter, and aggregate ``records`` into a contract panel.

        A text-typed ``value_col`` (REST payloads and warehouse TEXT columns
        arrive as strings) is coerced to float before aggregating; values
        that are not numbers raise :class:`DataContractError`. ``agg="count"``
        counts rows, so it never reads the values and never coerces them.
        """
        m = replace(self, **{k: v for k, v in overrides.items() if hasattr(self, k)})
        unknown = set(overrides) - set(self.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown mapping override(s) {sorted(unknown)}")
        if not isinstance(records, pd.DataFrame):
            raise DataContractError(
                f"expected a DataFrame of records, got {type(records).__name__}"
            )
        out = records.rename(columns=m.renames)
        needed = [*m.id_cols, m.date_col] + ([m.value_col] if m.value_col else [])
        missing = [c for c in needed if c not in out.columns]
        if missing:
            raise DataContractError(
                f"mapping {m.name!r} needs column(s) {missing} after renames "
                f"{m.renames}; records have {sorted(out.columns.astype(str))[:12]}"
            )
        for col, allowed in m.filters.items():
            if col not in out.columns:
                raise DataContractError(
                    f"mapping {m.name!r} filters on column {col!r}, which is missing "
                    f"after renames {m.renames}; records have "
                    f"{sorted(out.columns.astype(str))[:12]}. If this export has no "
                    f"{col!r} column and every row should be kept, override with "
                    "filters={}"
                )
            values = allowed if isinstance(allowed, (list, tuple, set)) else (allowed,)
            out = out[out[col].isin(tuple(values))]
        if len(out) == 0:
            raise DataContractError(
                f"mapping {m.name!r} matched no rows (filters: {m.filters})"
            )
        if m.value_col is not None and m.agg != "count":
            values = _numeric_values(out[m.value_col], m.name, m.value_col)
            out = out.assign(**{m.value_col: values})
        id_cols = list(m.id_cols) if m.id_cols else None
        if id_cols is None:
            out = out.assign(_series=m.name)
            id_cols = ["_series"]
        return to_panel(
            out,
            id_cols=id_cols,
            date_col=m.date_col,
            value_col=m.value_col,
            freq=m.freq,
            agg=m.agg,
            fill_value=m.fill_value,
            span=m.span,
            sep=m.sep,
            date_unit=m.date_unit,
            date_format=m.date_format,
        )


_MAPPINGS: dict[str, SchemaMapping] = {}


def register_mapping(mapping: SchemaMapping) -> SchemaMapping:
    """Register a mapping recipe under ``mapping.name`` (idempotent by value)."""
    existing = _MAPPINGS.get(mapping.name)
    if existing is not None and existing != mapping:
        raise ValueError(f"mapping name {mapping.name!r} already registered")
    _MAPPINGS[mapping.name] = mapping
    return mapping


def get_mapping(name: str) -> SchemaMapping:
    """Look up a registered mapping by name."""
    if name not in _MAPPINGS:
        available = ", ".join(sorted(_MAPPINGS)) or "<none>"
        raise ValueError(f"unknown mapping {name!r}; available: {available}")
    return _MAPPINGS[name]


def list_mappings() -> pd.DataFrame:
    """All registered mappings as a DataFrame (name, description, freq, agg)."""
    specs = sorted(_MAPPINGS.values(), key=lambda m: m.name)
    return pd.DataFrame(
        {
            "name": [m.name for m in specs],
            "description": [m.description for m in specs],
            "freq": [m.freq for m in specs],
            "agg": [m.agg for m in specs],
        }
    )


def apply_mapping(
    records: pd.DataFrame, mapping: SchemaMapping | str, **overrides
) -> pd.DataFrame:
    """Apply a mapping (instance or registered name) to raw records."""
    if isinstance(mapping, str):
        mapping = get_mapping(mapping)
    return mapping.apply(records, **overrides)


class Source(ABC):
    """A place records come from. Subclasses implement :meth:`fetch`.

    ``mapping`` (a :class:`SchemaMapping` or registered name) gives the source
    a default recipe so ``source.to_panel()`` is one call end to end.
    """

    mapping: SchemaMapping | str | None = None

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Return raw records as a DataFrame (one row per entity/event)."""

    def to_panel(self, mapping: SchemaMapping | str | None = None, **overrides) -> pd.DataFrame:
        """Fetch and shape into a contract panel in one step."""
        m = mapping or self.mapping
        if m is None:
            raise ValueError(
                f"{type(self).__name__} has no default mapping; pass mapping=..."
            )
        return apply_mapping(self.fetch(), m, **overrides)
