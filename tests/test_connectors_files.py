"""Tests for the file-backed sources (connectors.files).

CSVSource is exercised end to end on a messy export (quoted ``$1,234.50``
amounts); the numeric-text cleaning must stay conservative — genuinely-text
columns are left untouched. ParquetSource round-trips when a parquet engine
is installed and otherwise raises ImportError with an install hint.
"""

import pandas as pd
import pytest

from forecast_os.connectors import mappings  # noqa: F401  (registers the recipes)
from forecast_os.connectors.base import Source, apply_mapping
from forecast_os.connectors.files import CSVSource, ParquetSource, _clean_numeric_text
from forecast_os.core.exceptions import DataContractError
from forecast_os.core.types import ID_COL, TARGET_COL, TIME_COL


@pytest.fixture
def messy_csv(tmp_path):
    """HubSpot-style deal export with money-formatted amounts and text notes."""
    path = tmp_path / "deals.csv"
    path.write_text(
        "closedate,amount,dealstage,notes\n"
        '2026-01-05,"$1,234.50",closedwon,called Alice re: pricing\n'
        '2026-01-20,"$765.50",closedwon,sent proposal\n'
        '2026-01-22,"$9,999.00",closedlost,lost to competitor\n'
        '2026-02-14,"$2,500.00",closedwon,expansion deal\n'
    )
    return path


# -- CSVSource -----------------------------------------------------------------


def test_csv_source_stores_constructor_args_as_attributes(messy_csv):
    src = CSVSource(messy_csv, mapping="hubspot_deals", read_csv_kwargs={"sep": ","})
    assert src.path == messy_csv
    assert src.mapping == "hubspot_deals"
    assert src.read_csv_kwargs == {"sep": ","}
    assert isinstance(src, Source)


def test_csv_source_fetch_cleans_money_columns(messy_csv):
    records = CSVSource(messy_csv).fetch()
    assert list(records["amount"]) == [1234.50, 765.50, 9999.00, 2500.00]
    assert pd.api.types.is_float_dtype(records["amount"])
    # genuinely-text column untouched
    assert not pd.api.types.is_numeric_dtype(records["notes"])
    assert records["notes"].iloc[0] == "called Alice re: pricing"


def test_csv_source_to_panel_end_to_end(messy_csv):
    panel = CSVSource(messy_csv, mapping="hubspot_deals").to_panel()
    assert set(panel[ID_COL]) == {"hubspot_deals"}
    # Jan: 1234.50 + 765.50 closed-won (closedlost filtered); Feb: 2500.00
    assert list(panel[TARGET_COL]) == [2000.0, 2500.0]


def test_csv_source_to_panel_requires_a_mapping(messy_csv):
    with pytest.raises(ValueError, match="mapping"):
        CSVSource(messy_csv).to_panel()


def test_csv_source_read_csv_kwargs_passed_through(tmp_path):
    path = tmp_path / "semi.csv"
    path.write_text("date;whatever\n2026-01-01;a\n2026-01-01;b\n")
    src = CSVSource(path, mapping="generic_events", read_csv_kwargs={"sep": ";"})
    records = src.fetch()
    assert list(records.columns) == ["date", "whatever"]
    panel = src.to_panel()
    assert list(panel[TARGET_COL]) == [2.0]


# -- text-typed value columns (regression: strings were concatenated) ----------
#
# CSV/Parquet run _clean_numeric_text on fetch, but REST and SQL sources do
# not: HubSpot's CRM API returns every deal property as a JSON string and a
# warehouse TEXT column arrives the same way. Those frames reach
# SchemaMapping.apply as text, where summing CONCATENATED it — two $1,000 /
# $2,000 deals aggregated to 10002000.0 instead of 3000.0 — so the mapping
# itself must coerce the value column before aggregating.


@pytest.fixture
def text_deals():
    """HubSpot API-shaped records: every property value is a string."""
    return pd.DataFrame(
        {
            "closedate": ["2026-01-05", "2026-01-20", "2026-02-14"],
            "amount": ["1000", "2000", "500"],
            "dealstage": ["closedwon", "closedwon", "closedwon"],
        }
    )


def test_mapping_sums_text_typed_value_column(text_deals):
    """String amounts must be summed, not concatenated ('1000'+'2000' = 3000)."""
    panel = apply_mapping(text_deals, "hubspot_deals")
    assert list(panel[TARGET_COL]) == [3000.0, 500.0]


def test_mapping_means_text_typed_value_column(text_deals):
    """agg='mean' on a text column raised a bare TypeError instead of averaging."""
    panel = apply_mapping(text_deals, "hubspot_deals", agg="mean")
    assert list(panel[TARGET_COL]) == [1500.0, 500.0]


def test_mapping_strips_money_formatting_in_text_values(text_deals):
    """A text value column shares the file sources' money-format contract."""
    records = text_deals.assign(amount=["$1,234.50", "$765.50", "$2,500.00"])
    panel = apply_mapping(records, "hubspot_deals")
    assert list(panel[TARGET_COL]) == [2000.0, 2500.0]


def test_mapping_rejects_non_numeric_value_column(text_deals):
    """Genuinely non-numeric values must fail loudly, not produce a nonsense sum."""
    records = text_deals.assign(amount=["1000", "twelve hundred", "500"])
    with pytest.raises(DataContractError, match="twelve hundred"):
        apply_mapping(records, "hubspot_deals")


def test_mapping_treats_blank_text_values_as_missing(text_deals):
    """An unset property comes back as ''; that is a missing value, not an error."""
    records = text_deals.assign(amount=["1000", "", "500"])
    panel = apply_mapping(records, "hubspot_deals")
    # skipna, exactly as a NaN-valued row aggregates
    assert list(panel[TARGET_COL]) == [1000.0, 500.0]


def test_mapping_leaves_numeric_value_columns_alone(text_deals):
    """Already-numeric frames (REST JSON numbers, float CSVs) are untouched."""
    records = text_deals.assign(amount=[1000.5, 2000.25, 500.0])
    panel = apply_mapping(records, "hubspot_deals")
    assert list(panel[TARGET_COL]) == [3000.75, 500.0]


def test_mapping_rejects_european_formatted_money(text_deals):
    """'1.000,50' is one thousand and fifty cents, not 1.0005.

    The text coercion stripped ',' from every value with no shape check, so a
    de-DE/fr-FR export silently became 1000x too small (3.50 instead of
    3500.50) where it used to fail loudly. ``files._clean_numeric_text``
    validates against a money/number pattern before stripping and leaves
    anything else alone; this must share that pattern rather than guess.
    """
    records = text_deals.assign(amount=["1.000,50", "2.500,00", "500"])
    with pytest.raises(DataContractError, match=r"1\.000,50"):
        apply_mapping(records, "hubspot_deals")


def test_mapping_reads_accounting_negatives(text_deals):
    """'(500)' is the Excel/accounting spelling of -500, not a parse error."""
    records = text_deals.assign(amount=["(500)", "$(250.50)", "1,000"])
    panel = apply_mapping(records, "hubspot_deals")
    assert list(panel[TARGET_COL]) == [-750.5, 1000.0]


def test_mapping_count_agg_does_not_require_numeric_values(text_deals):
    """agg='count' counts rows, so a text value column is not an error.

    to_panel's count branch is a dtype-agnostic non-null count, but the new
    coercion ran unconditionally and rejected the stage labels it never
    reads. Blank strings must keep counting too: '' is a value that was
    present, and count asks how many rows there were.
    """
    records = text_deals.assign(amount=["won", "", "lost"])
    panel = apply_mapping(records, "hubspot_deals", agg="count")
    assert list(panel[TARGET_COL]) == [2.0, 1.0]


# -- GA4 CSV end to end (regression: YYYYMMDD dates must not become 1970) ------


@pytest.fixture
def ga4_csv(tmp_path):
    """GA4 BigQuery-style export: event_date is a YYYYMMDD day."""
    path = tmp_path / "ga4.csv"
    path.write_text(
        "event_date,event_name,user_pseudo_id\n"
        "20260101,page_view,a.1\n"
        "20260101,page_view,b.2\n"
        "20260101,purchase,a.1\n"
        "20260102,page_view,c.3\n"
    )
    return path


def _assert_ga4_panel_2026(panel):
    pv = panel[panel[ID_COL] == "page_view"]
    assert list(pv[TIME_COL]) == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")]
    assert list(pv[TARGET_COL]) == [2.0, 1.0]
    assert list(panel[panel[ID_COL] == "purchase"][TARGET_COL]) == [1.0]
    assert (panel[TIME_COL].dt.year == 2026).all()


def test_ga4_csv_to_panel_default_dtypes(ga4_csv):
    """read_csv infers event_date as int64; the recipe's date_format parses it."""
    _assert_ga4_panel_2026(CSVSource(ga4_csv, mapping="ga4_events").to_panel())


def test_ga4_csv_to_panel_with_dtype_str(ga4_csv):
    """With dtype=str the cleaner must NOT coerce 20260101 into float
    20260101.0 — 8-digit date-shaped columns are exempt from conversion."""
    src = CSVSource(ga4_csv, mapping="ga4_events", read_csv_kwargs={"dtype": str})
    _assert_ga4_panel_2026(src.to_panel())


# -- numeric-text cleaning -----------------------------------------------------


def test_cleaning_exempts_date_shaped_columns():
    """Regression: a column whose non-null values are all 8-digit strings is
    date-shaped (YYYYMMDD) and must stay text for date parsing."""
    records = pd.DataFrame(
        {
            "event_date": ["20260101", "20260102", None, "20260103"],
            "money": ["$1,000", "$2,000", "$3,000", "$4,000"],
        }
    )
    out = _clean_numeric_text(records)
    assert not pd.api.types.is_numeric_dtype(out["event_date"])
    assert list(out["event_date"].dropna()) == ["20260101", "20260102", "20260103"]
    assert list(out["money"]) == [1000.0, 2000.0, 3000.0, 4000.0]


def test_cleaning_still_converts_mixed_width_numeric_columns():
    # not ALL values are 8 digits, so the date-shape exemption does not apply
    records = pd.DataFrame({"amount": ["20260101", "1234", "500", "42", "7"]})
    out = _clean_numeric_text(records)
    assert pd.api.types.is_numeric_dtype(out["amount"])


def test_cleaning_is_conservative_on_mixed_columns():
    records = pd.DataFrame(
        {
            "ref": ["INV-1001", "INV-1002", "INV-1003", "INV-1004", "INV-1005"],
            # 3 of 5 look numeric (60% <= 80% threshold): untouched
            "mixed": ["$100", "$200", "300", "pending", "waived"],
            # exactly 80% is NOT more than 80%: untouched
            "borderline": ["1", "2", "3", "4", "n/a"],
            "money": ["$1,000", "$2,000", "$3,000", "$4,000", "$5,000"],
        }
    )
    out = _clean_numeric_text(records)
    for col in ("ref", "mixed", "borderline"):
        assert not pd.api.types.is_numeric_dtype(out[col]), col
        assert list(out[col]) == list(records[col]), col
    assert list(out["money"]) == [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]


def test_cleaning_handles_signs_nan_and_coerces_minority():
    records = pd.DataFrame(
        {
            # 5 of 6 non-null values look numeric (> 80%): the column converts
            # and the lone text value is coerced to NaN
            "amount": ["-$1,500.25", "$300", "2,000", "750.5", "1200", None, "refund"],
        }
    )
    out = _clean_numeric_text(records)
    assert list(out["amount"][:5]) == [-1500.25, 300.0, 2000.0, 750.5, 1200.0]
    assert out["amount"].isna().tolist() == [False] * 5 + [True, True]


def test_cleaning_leaves_dates_and_non_object_columns_alone():
    records = pd.DataFrame(
        {
            "closedate": ["2026-01-05", "2026-01-20", "2026-02-14"],
            "already_numeric": [1.5, 2.5, 3.5],
        }
    )
    out = _clean_numeric_text(records)
    assert not pd.api.types.is_numeric_dtype(out["closedate"])
    assert list(out["closedate"]) == list(records["closedate"])
    assert list(out["already_numeric"]) == [1.5, 2.5, 3.5]


def test_cleaning_does_not_mutate_input(messy_csv):
    records = pd.read_csv(messy_csv)
    _ = _clean_numeric_text(records)
    # the input frame keeps its raw text values
    assert not pd.api.types.is_numeric_dtype(records["amount"])
    assert records["amount"].iloc[0] == "$1,234.50"


# -- ParquetSource -------------------------------------------------------------


def test_parquet_source_stores_constructor_args_as_attributes():
    src = ParquetSource("deals.parquet", mapping="generic_events")
    assert src.path == "deals.parquet"
    assert src.mapping == "generic_events"
    assert isinstance(src, Source)


def test_parquet_source_missing_engine_hint(tmp_path, monkeypatch):
    def _no_engine(*args, **kwargs):
        raise ImportError("Unable to find a usable engine")

    monkeypatch.setattr(pd, "read_parquet", _no_engine)
    with pytest.raises(ImportError, match="pip install pyarrow"):
        ParquetSource(tmp_path / "x.parquet").fetch()


def test_parquet_source_roundtrip():
    pytest.importorskip("pyarrow")
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "deals.parquet"
        pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-01", "2026-01-02"],
                "amount_text": ["$1,000.00", "$250.00", "$99.50"],
            }
        ).to_parquet(path)
        records = ParquetSource(path).fetch()
        # strings survive the roundtrip; money-formatted text is cleaned
        assert list(records["date"]) == ["2026-01-01", "2026-01-01", "2026-01-02"]
        assert list(records["amount_text"]) == [1000.0, 250.0, 99.5]
        panel = ParquetSource(path, mapping="generic_events").to_panel()
        assert list(panel[TARGET_COL]) == [2.0, 1.0]
