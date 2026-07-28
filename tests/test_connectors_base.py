"""Tests for the connector contract itself (connectors.base).

The built-in recipes are the realistic vehicle here — they are what makes
multi-variant renames and ``id_cols`` overrides reachable from user code —
but every behaviour under test belongs to :class:`SchemaMapping.apply`.
"""

import pandas as pd
import pytest

from forecast_os.connectors import mappings  # noqa: F401  (registers the recipes)
from forecast_os.connectors.base import SchemaMapping, apply_mapping
from forecast_os.core.exceptions import DataContractError
from forecast_os.core.types import ID_COL, TARGET_COL


@pytest.fixture
def deals():
    """A small CRM export with an owner column."""
    return pd.DataFrame(
        {
            "close_date": ["2026-01-05", "2026-01-20", "2026-02-14"],
            "amount": [100.0, 200.0, 300.0],
            "owner": ["ann", "bob", "ann"],
            "stage": ["closedwon"] * 3,
        }
    )


# -- id_cols given as a bare string --------------------------------------------


def test_apply_accepts_a_bare_string_id_cols(deals):
    """Regression: ``apply(records, id_cols="owner")`` exploded the string
    into ['o','w','n','e','r'] and raised "needs column(s) ['o','w','n','e','r']".

    ``to_panel`` — the function ``apply`` wraps, and whose arguments the
    class docstring promises can all be overridden at apply time —
    normalises a bare string to a one-element list. ``apply`` must match it
    rather than iterate the string.
    """
    m = SchemaMapping(
        name="t", description="d", date_col="close_date", value_col="amount"
    )
    panel = m.apply(deals, id_cols="owner")
    assert set(panel[ID_COL]) == {"ann", "bob"}
    assert list(panel[panel[ID_COL] == "bob"][TARGET_COL]) == [200.0]


def test_bare_string_and_one_tuple_id_cols_agree(deals):
    m = SchemaMapping(
        name="t", description="d", date_col="close_date", value_col="amount"
    )
    pd.testing.assert_frame_equal(m.apply(deals, id_cols="owner"),
                                  m.apply(deals, id_cols=("owner",)))


def test_bare_string_id_cols_on_the_mapping_itself(deals):
    m = SchemaMapping(
        name="t",
        description="d",
        date_col="close_date",
        value_col="amount",
        id_cols="owner",
    )
    assert set(m.apply(deals)[ID_COL]) == {"ann", "bob"}


# -- rename collisions ---------------------------------------------------------
#
# Recipes list several source variants per canonical name ("date" AND
# "timestamp" -> "ts") on the premise that renaming an absent column is a
# no-op — sound only while at most one variant is present. When both are
# present pandas failed deep inside the shaping step ("cannot assemble with
# duplicate keys", or a bare "cannot reindex on an axis with duplicate
# labels" escaping unwrapped), naming neither the mapping nor the columns
# that collided.


def test_date_col_collision_names_both_source_columns():
    records = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "timestamp": ["2026-01-01T09:00:00", "2026-01-02T09:00:00"],
            "user": ["u1", "u2"],
        }
    )
    with pytest.raises(DataContractError, match="generic_events") as excinfo:
        apply_mapping(records, "generic_events")
    message = str(excinfo.value)
    assert "'ts'" in message
    assert "date" in message and "timestamp" in message


def test_filter_col_collision_raises_a_contract_error():
    """Both the report label ``Stage`` and the API field ``StageName``
    present used to escape as a raw pandas ValueError from the filter step."""
    records = pd.DataFrame(
        {
            "Close Date": ["2026-01-15"],
            "Amount": [100.0],
            "Stage": ["Closed Won"],
            "StageName": ["Closed Won"],
        }
    )
    with pytest.raises(DataContractError, match="salesforce_opportunities") as excinfo:
        apply_mapping(records, "salesforce_opportunities")
    assert "'stage'" in str(excinfo.value)


def test_value_col_collision_raises_a_contract_error():
    records = pd.DataFrame(
        {
            "CloseDate": ["2026-01-15"],
            "Amount": [100.0],
            "amount": [999.0],
            "StageName": ["Closed Won"],
        }
    )
    with pytest.raises(DataContractError, match="'amount'"):
        apply_mapping(records, "salesforce_opportunities")


def test_id_col_collision_raises_a_contract_error():
    records = pd.DataFrame(
        {
            "closedate": ["2026-01-15"],
            "amount": [100.0],
            "dealstage": ["closedwon"],
            "hubspot_owner_id": ["42"],
            "owner": ["ann"],
        }
    )
    with pytest.raises(DataContractError, match="'owner'"):
        apply_mapping(records, "hubspot_deals", id_cols=("owner",))


def test_collision_message_points_at_the_renames_override():
    records = pd.DataFrame(
        {"date": ["2026-01-01"], "timestamp": ["2026-01-01T09:00:00"]}
    )
    with pytest.raises(DataContractError) as excinfo:
        apply_mapping(records, "generic_events")
    assert "renames=" in str(excinfo.value)
    # and the documented escape hatch actually resolves it
    panel = apply_mapping(records, "generic_events", renames={"timestamp": "ts"})
    assert list(panel[TARGET_COL]) == [1.0]


def test_duplicate_columns_the_mapping_never_reads_are_left_alone():
    """The check must not turn ordinary messy exports into errors: only the
    columns the mapping actually reads can collide meaningfully."""
    records = pd.DataFrame(
        [["2026-01-01", "a", "b"], ["2026-01-02", "c", "d"]],
        columns=["date", "notes", "notes"],
    )
    panel = apply_mapping(records, "generic_events")
    assert list(panel[TARGET_COL]) == [1.0, 1.0]
