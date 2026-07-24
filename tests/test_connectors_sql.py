"""Tests for the SQL/warehouse source (connectors.sql.SQLSource).

All database behavior is exercised against stdlib in-memory sqlite3 — the
same DBAPI path pandas uses for any user-supplied driver connection.
"""

import sqlite3
from dataclasses import replace

import pandas as pd
import pytest

from forecast_os.connectors.base import SchemaMapping, register_mapping
from forecast_os.connectors.sql import SQLSource
from forecast_os.core.exceptions import ForecastOSError
from forecast_os.core.types import ID_COL, TARGET_COL, TIME_COL, validate_panel

DEALS_MAPPING = SchemaMapping(
    name="sqlite_deals_test",
    description="test recipe: sqlite deal rows to a monthly per-rep panel",
    date_col="close_date",
    id_cols=("rep",),
    value_col="amount",
    filters={"stage": ("closed_won",)},
    freq="MS",
)

register_mapping(
    replace(
        DEALS_MAPPING,
        name="sqlite_deals_won_test",
        description="registered variant of the sqlite deals test recipe",
    )
)


@pytest.fixture
def con():
    """In-memory deals table: 4 closed-won rows plus 1 closed-lost decoy."""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE deals (rep TEXT, close_date TEXT, amount REAL, stage TEXT)")
    con.executemany(
        "INSERT INTO deals VALUES (?, ?, ?, ?)",
        [
            ("alice", "2024-01-15", 100.0, "closed_won"),
            ("alice", "2024-01-20", 200.0, "closed_won"),
            ("alice", "2024-03-05", 50.0, "closed_won"),
            ("bob", "2024-02-10", 75.0, "closed_won"),
            ("bob", "2024-02-15", 999.0, "closed_lost"),
        ],
    )
    con.commit()
    yield con
    con.close()


class TestFetch:
    def test_fetch_returns_records_frame(self, con):
        frame = SQLSource("SELECT * FROM deals", con).fetch()
        assert isinstance(frame, pd.DataFrame)
        assert list(frame.columns) == ["rep", "close_date", "amount", "stage"]
        assert len(frame) == 5

    def test_constructor_args_stored_as_same_named_attributes(self, con):
        src = SQLSource(
            "SELECT * FROM deals", con, mapping=DEALS_MAPPING, params=("closed_won",)
        )
        assert src.query == "SELECT * FROM deals"
        assert src.con is con
        assert src.mapping is DEALS_MAPPING
        assert src.params == ("closed_won",)

    def test_params_default_to_none(self, con):
        assert SQLSource("SELECT * FROM deals", con).params is None

    def test_qmark_params_binding(self, con):
        src = SQLSource(
            "SELECT * FROM deals WHERE stage = ?", con, params=("closed_won",)
        )
        frame = src.fetch()
        assert len(frame) == 4
        assert set(frame["stage"]) == {"closed_won"}

    def test_named_params_binding(self, con):
        src = SQLSource("SELECT * FROM deals WHERE rep = :rep", con, params={"rep": "bob"})
        frame = src.fetch()
        assert len(frame) == 2
        assert set(frame["rep"]) == {"bob"}


class TestToPanel:
    def test_default_mapping_to_contract_panel(self, con):
        panel = SQLSource("SELECT * FROM deals", con, mapping=DEALS_MAPPING).to_panel()
        validate_panel(panel)
        assert list(panel.columns) == [ID_COL, TIME_COL, TARGET_COL]

        alice = panel[panel[ID_COL] == "alice"]
        # Jan = 100 + 200, Feb = gap fill, Mar = 50
        assert list(alice[TIME_COL]) == [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-02-01"),
            pd.Timestamp("2024-03-01"),
        ]
        assert list(alice[TARGET_COL]) == [300.0, 0.0, 50.0]

        bob = panel[panel[ID_COL] == "bob"]
        # the closed_lost row is dropped by the mapping's stage filter
        assert list(bob[TARGET_COL]) == [75.0]

    def test_registered_mapping_name(self, con):
        panel = SQLSource("SELECT * FROM deals", con, mapping="sqlite_deals_won_test").to_panel()
        alice = panel[panel[ID_COL] == "alice"]
        assert list(alice[TARGET_COL]) == [300.0, 0.0, 50.0]

    def test_call_time_mapping_and_override(self, con):
        src = SQLSource("SELECT * FROM deals", con)
        panel = src.to_panel(mapping=DEALS_MAPPING, freq="QS")
        alice = panel[panel[ID_COL] == "alice"]
        # all three closed-won alice deals land in Q1 2024
        assert list(alice[TIME_COL]) == [pd.Timestamp("2024-01-01")]
        assert list(alice[TARGET_COL]) == [350.0]

    def test_mapping_renames_compose_with_sql_aliases(self, con):
        aliased = replace(
            DEALS_MAPPING,
            name="sqlite_deals_aliased_test",
            renames={"deal_owner": "rep"},
        )
        src = SQLSource(
            "SELECT rep AS deal_owner, close_date, amount, stage FROM deals",
            con,
            mapping=aliased,
        )
        panel = src.to_panel()
        assert set(panel[ID_COL]) == {"alice", "bob"}

    def test_missing_mapping_raises(self, con):
        with pytest.raises(ValueError, match="no default mapping"):
            SQLSource("SELECT * FROM deals", con).to_panel()


class TestErrorWrapping:
    def test_bad_query_raises_forecast_os_error_with_query(self, con):
        src = SQLSource("SELECT * FROM missing_table", con)
        with pytest.raises(ForecastOSError, match="missing_table") as excinfo:
            src.fetch()
        # the driver/pandas error stays on the chain for debugging
        assert excinfo.value.__cause__ is not None

    def test_long_query_is_truncated_in_message(self, con):
        query = "SELECT * FROM missing_table WHERE " + "x = 1 AND " * 60 + "x = 1"
        with pytest.raises(ForecastOSError) as excinfo:
            SQLSource(query, con).fetch()
        msg = str(excinfo.value)
        assert "missing_table" in msg
        assert "..." in msg
        # the full query text must never be echoed verbatim
        assert query not in msg

    def test_bad_params_are_wrapped(self, con):
        src = SQLSource("SELECT * FROM deals WHERE rep = ?", con, params=())
        with pytest.raises(ForecastOSError, match="query"):
            src.fetch()
