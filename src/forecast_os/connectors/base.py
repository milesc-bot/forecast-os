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
        """Rename, filter, and aggregate ``records`` into a contract panel."""
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
