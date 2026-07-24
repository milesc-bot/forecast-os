"""SQL/warehouse source: forecast straight from a query, any driver.

:class:`SQLSource` runs a query through :func:`pandas.read_sql`, so ``con``
is anything pandas accepts — a DBAPI connection, a SQLAlchemy engine, or a
connection string. The USER brings the driver, which is how Snowflake,
BigQuery, Postgres, and DuckDB all work with zero new dependencies here:
install the vendor package, open a connection, hand it over.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from ..core.exceptions import ForecastOSError
from .base import SchemaMapping, Source

__all__ = ["SQLSource"]

#: Longest slice of the failing query echoed into error messages.
_QUERY_PREVIEW_CHARS = 200


def _preview(query: str, limit: int = _QUERY_PREVIEW_CHARS) -> str:
    """The query collapsed to one line and truncated to ``limit`` characters."""
    flat = " ".join(str(query).split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


class SQLSource(Source):
    """Records from a SQL query against any pandas-readable connection.

    Parameters
    ----------
    query : SQL text returning one row per record (deal, invoice, event).
    con : anything :func:`pandas.read_sql` accepts — a DBAPI connection
        (sqlite3, duckdb, snowflake-connector, ...), a SQLAlchemy engine or
        connectable, or a database URI string.
    mapping : default :class:`~forecast_os.connectors.base.SchemaMapping`
        (or registered mapping name) so :meth:`to_panel` is one call.
    params : query parameters forwarded to the driver — a sequence for
        positional styles (``?``/``%s``) or a mapping for named styles
        (``:name``).

    Examples
    --------
    DuckDB (``pip install duckdb``)::

        import duckdb

        src = SQLSource(
            "SELECT rep, close_date, amount FROM deals WHERE stage = ?",
            duckdb.connect("crm.duckdb"),
            mapping=deals_mapping,
            params=("closed_won",),
        )
        panel = src.to_panel()

    Snowflake (``pip install snowflake-connector-python``)::

        import snowflake.connector

        con = snowflake.connector.connect(account="...", user="...", password="...")
        src = SQLSource(
            "SELECT owner, close_date, amount FROM analytics.opportunities",
            con,
        )
        panel = src.to_panel(mapping="my_warehouse_deals", freq="W")

    Database errors are wrapped in
    :class:`~forecast_os.core.exceptions.ForecastOSError` carrying the
    (truncated) query, with the driver exception preserved as ``__cause__``.
    """

    def __init__(
        self,
        query: str,
        con: Any,
        mapping: SchemaMapping | str | None = None,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
    ):
        self.query = query
        self.con = con
        self.mapping = mapping
        self.params = params

    def fetch(self) -> pd.DataFrame:
        """Run the query and return the result as a records DataFrame."""
        try:
            return pd.read_sql(self.query, self.con, params=self.params)
        except Exception as exc:
            # the driver message is truncated too: pandas embeds the full SQL
            # text in its DatabaseError, which would defeat the query preview
            raise ForecastOSError(
                f"SQL query failed ({type(exc).__name__}: {_preview(str(exc))}); "
                f"query: {_preview(self.query)}"
            ) from exc
