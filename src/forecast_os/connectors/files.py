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

#: An identifier-shaped value: a whole number written with a leading zero
#: ("00190", "0190"). The padding carries meaning — GL/account codes, cost
#: centres, zip codes and SKUs are identifiers, not quantities — and
#: converting them merges "00190"/"0190"/"190" into one series. Matched
#: against the value the conversion would see (sign/currency/whitespace
#: already stripped), so "-0190" and "$0190" are exempt too.
_ZERO_PADDED_RE = re.compile(r"0\d+")


def _clean_numeric_text(records: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns that read as money/numbers into floats (a copy).

    Conservative by design: a column is converted only when strictly more
    than 80% of its non-null values matches a money/number pattern
    (optional sign, optional currency symbol, thousands separators,
    decimals), so genuinely-text columns — and columns that are only partly
    numeric — are left untouched. Columns whose non-null values are ALL
    exactly 8 digits are date-shaped (``YYYYMMDD``, e.g. GA4 ``event_date``)
    and are exempt from conversion so they reach date parsing as text. A
    column holding ANY zero-padded whole number (``"00190"``, and likewise
    ``"-0190"``/``"$0190"``) is exempt for the same reason: the padding is
    significant, so the value is an identifier rather than a quantity. In a
    converted column the minority of values that do not parse become NaN
    (``pd.to_numeric(errors="coerce")``).

    Both exemptions only reach columns that arrive here as text, which is
    the whole story for Parquet but not for CSV: ``pandas.read_csv`` parses
    an all-digit column to int64 (dropping the padding) before this
    function ever sees it, so a CSV whose identifier or ``YYYYMMDD`` column
    must survive has to be read with ``read_csv_kwargs={"dtype": str}``.
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
        # test the padding on what the conversion would consume, not the raw
        # text: "0190" was exempt while "-0190"/"$0190"/" 0190" collapsed
        bare = values.str.replace(_STRIP_RE, "", regex=True).str.lstrip("+-")
        if bare.str.fullmatch(_ZERO_PADDED_RE).any():
            continue  # zero-padded identifier ("00190"): the padding matters
        stripped = s.where(s.isna(), s.astype(str).str.replace(_STRIP_RE, "", regex=True))
        out[col] = pd.to_numeric(stripped, errors="coerce")
    return out


class CSVSource(Source):
    """Records from a CSV export on disk.

    ``mapping`` (a :class:`~forecast_os.connectors.base.SchemaMapping` or a
    registered recipe name) becomes the source's default recipe for
    :meth:`~forecast_os.connectors.base.Source.to_panel`. ``read_csv_kwargs``
    are passed through to :func:`pandas.read_csv` (e.g. ``{"sep": ";"}``).

    Pass ``read_csv_kwargs={"dtype": str}`` when a column's text form
    matters — zero-padded identifiers (GL/account codes, zip codes, SKUs)
    and ``YYYYMMDD`` dates. ``read_csv`` parses those to int64 on its own,
    which drops the padding before the numeric-text cleaner (which exempts
    them) can see them.
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
