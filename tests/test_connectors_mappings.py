"""Tests for the built-in platform mapping recipes (connectors.mappings).

Each recipe is applied to a synthetic export frame built with the platform's
REAL export column names; aggregated values (including status filters and
gap fill) are hand-checked.
"""

import pandas as pd
import pytest

from forecast_os.connectors import mappings  # noqa: F401  (registers the recipes)
from forecast_os.connectors.base import apply_mapping, get_mapping, list_mappings
from forecast_os.core.exceptions import DataContractError
from forecast_os.core.types import ID_COL, TARGET_COL, TIME_COL, validate_panel


def _epoch(day: str) -> int:
    """Unix epoch seconds for a wall-clock day (naive Timestamp = UTC)."""
    return int(pd.Timestamp(day).timestamp())


EXPECTED_RECIPES = {
    "salesforce_opportunities",
    "hubspot_deals",
    "pipedrive_deals",
    "stripe_invoices",
    "stripe_charges",
    "posthog_events",
    "ga4_events",
    "mixpanel_events",
    "amplitude_events",
    "generic_events",
}


# -- registry ------------------------------------------------------------------


def test_all_expected_recipes_registered_with_descriptions():
    listing = list_mappings()
    assert EXPECTED_RECIPES <= set(listing["name"])
    described = listing[listing["name"].isin(EXPECTED_RECIPES)]
    assert (described["description"].str.len() > 0).all()


def test_event_recipes_are_daily_counts():
    for name in ("posthog_events", "ga4_events", "mixpanel_events",
                 "amplitude_events", "generic_events"):
        m = get_mapping(name)
        assert m.freq == "D", name
        assert m.value_col is None, name


def test_stripe_descriptions_note_minor_units():
    for name in ("stripe_invoices", "stripe_charges"):
        description = get_mapping(name).description.lower()
        assert "minor units" in description or "cents" in description


def test_epoch_second_recipes_declare_date_unit():
    """Stripe created / Mixpanel time are Unix epoch seconds in real exports."""
    for name in ("stripe_invoices", "stripe_charges", "mixpanel_events"):
        assert get_mapping(name).date_unit == "s", name


def test_ga4_recipe_is_single_variant_event_date():
    """ga4_events parses event_date (YYYYMMDD) only; event_timestamp was
    dropped from renames because the two variants need different parsing."""
    m = get_mapping("ga4_events")
    assert m.date_format == "%Y%m%d"
    assert "event_timestamp" not in m.renames
    assert "event_date" in m.description


# -- CRM / billing recipes (value sums with status filters) --------------------


def _salesforce_report():
    """Salesforce opportunity report export (UI report column labels)."""
    return pd.DataFrame(
        {
            "Opportunity Name": ["Acme 100", "Beta Expansion", "Gamma", "Delta", "Eps"],
            "Account Name": ["Acme", "Beta", "Gamma", "Delta", "Eps"],
            "Opportunity Owner": ["alice", "bob", "alice", "bob", "alice"],
            "Stage": ["Closed Won", "Closed Won", "Closed Lost", "Negotiation", "Closed Won"],
            "Amount": [1000.0, 2000.0, 5000.0, 700.0, 1500.0],
            "Close Date": ["2026-01-15", "2026-01-20", "2026-01-22", "2026-02-10", "2026-03-05"],
        }
    )


def test_salesforce_opportunities_report_export():
    panel = apply_mapping(_salesforce_report(), "salesforce_opportunities")
    validate_panel(panel)
    # single series named after the mapping; only Closed Won kept
    assert set(panel[ID_COL]) == {"salesforce_opportunities"}
    assert list(panel[TIME_COL]) == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-02-01"),  # no closed-won deals: gap filled with 0
        pd.Timestamp("2026-03-01"),
    ]
    assert list(panel[TARGET_COL]) == [3000.0, 0.0, 1500.0]


def test_salesforce_opportunities_api_column_variants():
    api = pd.DataFrame(
        {
            "CloseDate": ["2026-01-15", "2026-01-20", "2026-02-11"],
            "StageName": ["Closed Won", "Closed Lost", "Closed Won"],
            "Amount": [100.0, 900.0, 50.0],
            "Owner": ["alice", "alice", "bob"],
        }
    )
    panel = apply_mapping(api, "salesforce_opportunities")
    assert list(panel[TARGET_COL]) == [100.0, 50.0]


def test_salesforce_opportunities_owner_id_cols_override():
    panel = apply_mapping(_salesforce_report(), "salesforce_opportunities", id_cols=("owner",))
    alice = panel[panel[ID_COL] == "alice"]
    bob = panel[panel[ID_COL] == "bob"]
    # alice: Jan 1000, Feb gap -> 0, Mar 1500; bob: Jan 2000 only
    assert list(alice[TARGET_COL]) == [1000.0, 0.0, 1500.0]
    assert list(bob[TARGET_COL]) == [2000.0]


def test_hubspot_deals_export():
    records = pd.DataFrame(
        {
            "dealname": ["A", "B", "C", "D"],
            "amount": [500.0, 250.0, 900.0, 300.0],
            "closedate": ["2026-01-05", "2026-01-20", "2026-01-22", "2026-02-10"],
            "dealstage": ["closedwon", "closedwon", "closedlost", "closedwon"],
            "hubspot_owner_id": ["42", "42", "7", "7"],
            "pipeline": ["default"] * 4,
        }
    )
    panel = apply_mapping(records, "hubspot_deals")
    assert set(panel[ID_COL]) == {"hubspot_deals"}
    assert list(panel[TARGET_COL]) == [750.0, 300.0]
    # hubspot_owner_id is pre-renamed to owner so an id_cols override just works
    by_owner = apply_mapping(records, "hubspot_deals", id_cols=("owner",))
    assert set(by_owner[ID_COL]) == {"42", "7"}
    assert list(by_owner[by_owner[ID_COL] == "42"][TARGET_COL]) == [750.0]


def test_pipedrive_deals_export():
    records = pd.DataFrame(
        {
            "title": ["A", "B", "C", "D"],
            "value": [100.0, 200.0, 400.0, 50.0],
            "currency": ["USD"] * 4,
            "status": ["won", "won", "lost", "won"],
            # a lost deal realistically has no won_time; the filter drops it
            "won_time": ["2026-01-03", "2026-01-28", None, "2026-02-15"],
            "owner_name": ["alice", "bob", "alice", "bob"],
        }
    )
    panel = apply_mapping(records, "pipedrive_deals")
    assert set(panel[ID_COL]) == {"pipedrive_deals"}
    assert list(panel[TARGET_COL]) == [300.0, 50.0]


def test_stripe_invoices_export_sums_minor_units():
    # real Stripe exports carry `created` as Unix epoch seconds (integers)
    records = pd.DataFrame(
        {
            "id": ["in_1", "in_2", "in_3", "in_4", "in_5"],
            "customer": ["cus_a", "cus_b", "cus_a", "cus_c", "cus_b"],
            "amount_due": [1999, 4999, 999, 500, 2999],
            "amount_paid": [1999, 4999, 0, 0, 2999],
            "currency": ["usd"] * 5,
            "status": ["paid", "paid", "open", "void", "paid"],
            "created": [
                _epoch("2026-01-04"), _epoch("2026-01-19"), _epoch("2026-01-25"),
                _epoch("2026-01-30"), _epoch("2026-02-08"),
            ],
        }
    )
    panel = apply_mapping(records, "stripe_invoices")
    # cents stay cents: 1999 + 4999 in Jan, 2999 in Feb
    assert list(panel[TARGET_COL]) == [6998.0, 2999.0]
    assert list(panel[TIME_COL]) == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01")]


def test_stripe_charges_export_keeps_succeeded_only():
    records = pd.DataFrame(
        {
            "id": ["ch_1", "ch_2", "ch_3", "ch_4"],
            "amount": [1000, 2000, 500, 750],
            "currency": ["usd"] * 4,
            "status": ["succeeded", "succeeded", "failed", "succeeded"],
            "paid": [True, True, False, True],
            "created": [
                _epoch("2026-01-10"), _epoch("2026-01-15"),
                _epoch("2026-01-16"), _epoch("2026-02-02"),
            ],
        }
    )
    panel = apply_mapping(records, "stripe_charges")
    assert list(panel[TARGET_COL]) == [3000.0, 750.0]
    assert list(panel[TIME_COL]) == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01")]


def test_stripe_epoch_seconds_bucket_2026_not_1970():
    """Regression: epoch-second `created` collapsed into a single 1970-01-01
    bucket (ints read as nanosecond offsets). Deals across Jan/Feb/Mar 2026
    must produce exactly those three monthly buckets."""
    records = pd.DataFrame(
        {
            "id": ["ch_1", "ch_2", "ch_3"],
            "amount": [1000, 2000, 4000],
            "status": ["succeeded"] * 3,
            "created": [_epoch("2026-01-15"), _epoch("2026-02-10"), _epoch("2026-03-05")],
        }
    )
    panel = apply_mapping(records, "stripe_charges")
    assert list(panel[TIME_COL]) == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-02-01"),
        pd.Timestamp("2026-03-01"),
    ]
    assert list(panel[TARGET_COL]) == [1000.0, 2000.0, 4000.0]


# -- product analytics recipes (daily event counts) ----------------------------


def test_posthog_events_daily_counts_per_event():
    records = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T09:00:00", "2026-01-01T10:30:00", "2026-01-01T11:00:00",
                "2026-01-02T08:15:00", "2026-01-03T09:45:00", "2026-01-03T22:00:00",
            ],
            "event": ["$pageview", "$pageview", "signup", "$pageview", "$pageview", "$pageview"],
            "distinct_id": ["u1", "u2", "u1", "u3", "u1", "u2"],
        }
    )
    panel = apply_mapping(records, "posthog_events")
    pv = panel[panel[ID_COL] == "$pageview"]
    assert list(pv[TIME_COL]) == list(pd.date_range("2026-01-01", periods=3, freq="D"))
    assert list(pv[TARGET_COL]) == [2.0, 1.0, 2.0]
    assert list(panel[panel[ID_COL] == "signup"][TARGET_COL]) == [1.0]


def test_ga4_events_daily_counts_from_event_date():
    records = pd.DataFrame(
        {
            "event_date": ["20260101", "20260101", "20260101", "20260102"],
            "event_name": ["page_view", "page_view", "purchase", "page_view"],
            "user_pseudo_id": ["a.1", "b.2", "a.1", "c.3"],
        }
    )
    panel = apply_mapping(records, "ga4_events")
    assert list(panel[panel[ID_COL] == "page_view"][TARGET_COL]) == [2.0, 1.0]
    assert list(panel[panel[ID_COL] == "purchase"][TARGET_COL]) == [1.0]
    assert panel[TIME_COL].min() == pd.Timestamp("2026-01-01")


def test_ga4_integer_event_date_parses_as_2026_not_1970():
    """Regression: BigQuery delivers event_date as YYYYMMDD integers, which
    to_datetime read as nanosecond epoch offsets (all 1970-01-01)."""
    records = pd.DataFrame(
        {
            "event_date": [20260115, 20260115, 20260116],
            "event_name": ["page_view", "purchase", "page_view"],
            "user_pseudo_id": ["a.1", "a.1", "b.2"],
        }
    )
    panel = apply_mapping(records, "ga4_events")
    pv = panel[panel[ID_COL] == "page_view"]
    assert list(pv[TIME_COL]) == [pd.Timestamp("2026-01-15"), pd.Timestamp("2026-01-16")]
    assert list(pv[TARGET_COL]) == [1.0, 1.0]
    assert (panel[TIME_COL].dt.year == 2026).all()


def test_mixpanel_events_daily_counts():
    # real Mixpanel exports carry `time` as Unix epoch seconds
    records = pd.DataFrame(
        {
            "event": ["Sign Up", "Sign Up", "Play Song", "Sign Up"],
            "time": [
                _epoch("2026-01-01 09:30:00"), _epoch("2026-01-01 17:00:00"),
                _epoch("2026-01-01 18:00:00"), _epoch("2026-01-02 10:00:00"),
            ],
            "distinct_id": ["u1", "u2", "u1", "u3"],
        }
    )
    panel = apply_mapping(records, "mixpanel_events")
    assert list(panel[panel[ID_COL] == "Sign Up"][TARGET_COL]) == [2.0, 1.0]
    assert list(panel[panel[ID_COL] == "Play Song"][TARGET_COL]) == [1.0]


def test_mixpanel_epoch_seconds_bucket_2026_days_not_1970():
    """Regression: epoch-second `time` collapsed every event into 1970-01-01;
    Jan/Feb/Mar 2026 events must land on their real 2026 days."""
    records = pd.DataFrame(
        {
            "event": ["Sign Up"] * 3,
            "time": [
                _epoch("2026-01-15 09:30:00"),
                _epoch("2026-02-10 17:00:00"),
                _epoch("2026-03-05 23:59:59"),
            ],
            "distinct_id": ["u1", "u2", "u3"],
        }
    )
    panel = apply_mapping(records, "mixpanel_events", fill_value=0.0)
    observed = panel[panel[TARGET_COL] > 0]
    assert list(observed[TIME_COL]) == [
        pd.Timestamp("2026-01-15"),
        pd.Timestamp("2026-02-10"),
        pd.Timestamp("2026-03-05"),
    ]


def test_amplitude_events_daily_counts():
    records = pd.DataFrame(
        {
            "event_time": [
                "2026-01-01 12:00:00.000", "2026-01-02 13:00:00.000",
                "2026-01-02 14:00:00.000", "2026-01-02 15:00:00.000",
            ],
            "event_type": ["start_session", "start_session", "start_session", "purchase"],
            "user_id": ["u1", "u1", "u2", "u2"],
        }
    )
    panel = apply_mapping(records, "amplitude_events")
    assert list(panel[panel[ID_COL] == "start_session"][TARGET_COL]) == [1.0, 2.0]
    assert list(panel[panel[ID_COL] == "purchase"][TARGET_COL]) == [1.0]


def test_generic_events_counts_all_rows_as_one_series():
    records = pd.DataFrame(
        {"date": ["2026-01-01", "2026-01-01", "2026-01-03"], "whatever": ["a", "b", "c"]}
    )
    panel = apply_mapping(records, "generic_events")
    assert set(panel[ID_COL]) == {"generic_events"}
    # interior day with no events is filled with 0
    assert list(panel[TARGET_COL]) == [2.0, 0.0, 1.0]


def test_generic_events_timestamp_variant_and_freq_override():
    records = pd.DataFrame(
        {"timestamp": ["2026-01-01 10:00:00", "2026-01-14 11:00:00", "2026-01-30 12:00:00"]}
    )
    panel = apply_mapping(records, "generic_events", freq="MS")
    assert list(panel[TIME_COL]) == [pd.Timestamp("2026-01-01")]
    assert list(panel[TARGET_COL]) == [3.0]


# -- errors --------------------------------------------------------------------


def test_missing_column_error_names_mapping_and_columns():
    records = pd.DataFrame(
        {"closedate": ["2026-01-05"], "dealstage": ["closedwon"]}  # no amount
    )
    with pytest.raises(DataContractError, match="hubspot_deals") as excinfo:
        apply_mapping(records, "hubspot_deals")
    assert "amount" in str(excinfo.value)


def test_missing_filter_column_raises_instead_of_silently_keeping_all_rows():
    """Regression: a missing filter column was silently skipped, so e.g. an
    export without a stage column summed lost deals into the panel."""
    records = pd.DataFrame(
        {"closedate": ["2026-01-05", "2026-01-20"], "amount": [100.0, 900.0]}  # no dealstage
    )
    with pytest.raises(DataContractError, match="stage") as excinfo:
        apply_mapping(records, "hubspot_deals")
    message = str(excinfo.value)
    assert "hubspot_deals" in message
    assert "filters={}" in message  # the documented way to disable filtering
    # the explicit opt-out keeps every row
    panel = apply_mapping(records, "hubspot_deals", filters={})
    assert list(panel[TARGET_COL]) == [1000.0]


def test_filters_remove_everything_raises():
    records = pd.DataFrame(
        {
            "closedate": ["2026-01-05"],
            "amount": [100.0],
            "dealstage": ["closedlost"],
        }
    )
    with pytest.raises(DataContractError, match="no rows"):
        apply_mapping(records, "hubspot_deals")


def test_unknown_mapping_name_lists_available():
    with pytest.raises(ValueError, match="unknown mapping"):
        apply_mapping(pd.DataFrame({"a": [1]}), "not_a_mapping")
