"""File-backed record sources: CSV and Parquet exports on disk.

Both sources read one-row-per-record frames and run a conservative
numeric-text cleaner so money-formatted export columns (``"$1,234.50"``)
arrive as floats::

    CSVSource("deals.csv", mapping="hubspot_deals").to_panel()
    ParquetSource("events.parquet", mapping="generic_events").to_panel()

Reading Parquet requires a pandas parquet engine (``pip install pyarrow``);
CSV needs nothing beyond pandas.
"""

from __future__ import annotations

import re
from os import PathLike

import pandas as pd

from .base import SchemaMapping, Source

__all__ = ["CSVSource", "ParquetSource"]

#: A money/number-formatted string: optional sign, optional currency symbol,
#: digits with optional thousands separators, optional decimals.
_NUMERIC_TEXT_RE = re.compile(
    r"^\s*[-+]?\s*[$€£¥]?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*$"
)
_STRIP_RE = re.compile(r"[$€£¥,\s]")

#: A text column converts only when strictly more than this share of its
#: non-null values matches the money/number pattern.
_NUMERIC_SHARE = 0.8

#: A date-shaped value: exactly 8 digits (YYYYMMDD, e.g. GA4 event_date).
#: Columns made entirely of these must stay text for date parsing —
#: converting 20260101 to float 20260101.0 corrupts the date column.
_DATE_SHAPED_RE = re.compile(r"\d{8}")


def _clean_numeric_text(records: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns that read as money/numbers into floats (a copy).

    Conservative by design: a column is converted only when strictly more
    than 80% of its non-null values matches a money/number pattern
    (optional sign, optional currency symbol, thousands separators,
    decimals), so genuinely-text columns — and columns that are only partly
    numeric — are left untouched. Columns whose non-null values are ALL
    exactly 8 digits are date-shaped (``YYYYMMDD``, e.g. GA4 ``event_date``)
    and are exempt from conversion so they reach date parsing as text. In a
    converted column the minority of values that do not parse become NaN
    (``pd.to_numeric(errors="coerce")``).
    """
    out = records.copy()
    for col in out.columns:
        s = out[col]
        if not (s.dtype == object or isinstance(s.dtype, pd.StringDtype)):
            continue
        values = s.dropna().astype(str)
        if len(values) == 0 or values.str.match(_NUMERIC_TEXT_RE).mean() <= _NUMERIC_SHARE:
            continue
        if values.str.fullmatch(_DATE_SHAPED_RE).all():
            continue  # date-shaped (YYYYMMDD): leave as text for date parsing
        stripped = s.where(s.isna(), s.astype(str).str.replace(_STRIP_RE, "", regex=True))
        out[col] = pd.to_numeric(stripped, errors="coerce")
    return out


class CSVSource(Source):
    """Records from a CSV export on disk.

    ``mapping`` (a :class:`~forecast_os.connectors.base.SchemaMapping` or a
    registered recipe name) becomes the source's default recipe for
    :meth:`~forecast_os.connectors.base.Source.to_panel`. ``read_csv_kwargs``
    are passed through to :func:`pandas.read_csv` (e.g. ``{"sep": ";"}``).
    """

    def __init__(
        self,
        path: str | PathLike,
        mapping: SchemaMapping | str | None = None,
        read_csv_kwargs: dict | None = None,
    ):
        self.path = path
        self.mapping = mapping
        self.read_csv_kwargs = read_csv_kwargs

    def fetch(self) -> pd.DataFrame:
        """Read the CSV and clean money-formatted numeric text columns."""
        records = pd.read_csv(self.path, **(self.read_csv_kwargs or {}))
        return _clean_numeric_text(records)


class ParquetSource(Source):
    """Records from a Parquet file on disk.

    ``mapping`` (a :class:`~forecast_os.connectors.base.SchemaMapping` or a
    registered recipe name) becomes the source's default recipe for
    :meth:`~forecast_os.connectors.base.Source.to_panel`. Reading Parquet
    needs a pandas parquet engine (pyarrow or fastparquet); without one,
    :meth:`fetch` raises ``ImportError`` with an install hint.
    """

    def __init__(self, path: str | PathLike, mapping: SchemaMapping | str | None = None):
        self.path = path
        self.mapping = mapping

    def fetch(self) -> pd.DataFrame:
        """Read the Parquet file and clean money-formatted numeric text columns."""
        try:
            records = pd.read_parquet(self.path)
        except ImportError as exc:
            raise ImportError(
                "reading Parquet requires a parquet engine: pip install pyarrow"
            ) from exc
        return _clean_numeric_text(records)
