"""Tests for the GTM event bridge (gtm.events.to_panel)."""

import numpy as np
import pandas as pd
import pytest

from forecast_os.core.exceptions import DataContractError
from forecast_os.core.types import ID_COL, TARGET_COL, TIME_COL, validate_panel
from forecast_os.gtm import to_panel


def _sfdc_export():
    """Salesforce-style opportunity export: one row per closed deal.

    Duplicate close dates are legal; alice skips February entirely (gap);
    ids are two-level (team, rep) so the panel is hierarchy-ready.
    """
    return pd.DataFrame(
        {
            "team": ["east", "east", "east", "west"],
            "rep": ["alice", "alice", "alice", "bob"],
            "close_date": ["2024-01-15", "2024-01-15", "2024-03-05", "2024-02-10"],
            "amount": [100.0, 200.0, 50.0, 75.0],
        }
    )


class TestToPanelHappyPath:
    def test_sum_gap_fill_and_hierarchy_ids(self):
        panel = to_panel(
            _sfdc_export(), id_cols=["team", "rep"], date_col="close_date", value_col="amount"
        )
        # contract-clean
        validate_panel(panel)
        assert list(panel.columns) == [ID_COL, TIME_COL, TARGET_COL]
        # hierarchy-ready ids joined by "/"
        assert set(panel[ID_COL]) == {"east/alice", "west/bob"}

        alice = panel[panel[ID_COL] == "east/alice"]
        # gap month (Feb) filled between first and last period
        assert list(alice[TIME_COL]) == [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-02-01"),
            pd.Timestamp("2024-03-01"),
        ]
        # hand-checked sums: Jan = 100 + 200 (duplicate dates), Feb = fill, Mar = 50
        assert list(alice[TARGET_COL]) == [300.0, 0.0, 50.0]

        bob = panel[panel[ID_COL] == "west/bob"]
        assert list(bob[TIME_COL]) == [pd.Timestamp("2024-02-01")]
        assert list(bob[TARGET_COL]) == [75.0]

    def test_count_when_value_col_is_none(self):
        panel = to_panel(_sfdc_export(), id_cols=["team", "rep"], date_col="close_date")
        alice = panel[panel[ID_COL] == "east/alice"]
        assert list(alice[TARGET_COL]) == [2.0, 0.0, 1.0]

    def test_explicit_count_agg(self):
        panel = to_panel(
            _sfdc_export(),
            id_cols=["team", "rep"],
            date_col="close_date",
            value_col="amount",
            agg="count",
        )
        alice = panel[panel[ID_COL] == "east/alice"]
        assert list(alice[TARGET_COL]) == [2.0, 0.0, 1.0]

    def test_mean_agg(self):
        panel = to_panel(
            _sfdc_export(),
            id_cols=["team", "rep"],
            date_col="close_date",
            value_col="amount",
            agg="mean",
        )
        alice = panel[panel[ID_COL] == "east/alice"]
        assert list(alice[TARGET_COL]) == [150.0, 0.0, 50.0]

    def test_single_id_col_as_string(self):
        panel = to_panel(_sfdc_export(), id_cols="rep", date_col="close_date")
        assert set(panel[ID_COL]) == {"alice", "bob"}

    def test_custom_sep(self):
        panel = to_panel(
            _sfdc_export(), id_cols=["team", "rep"], date_col="close_date", sep="|"
        )
        assert set(panel[ID_COL]) == {"east|alice", "west|bob"}

    def test_custom_fill_value(self):
        panel = to_panel(
            _sfdc_export(),
            id_cols=["team", "rep"],
            date_col="close_date",
            value_col="amount",
            fill_value=-1.0,
        )
        alice = panel[panel[ID_COL] == "east/alice"]
        assert alice[TARGET_COL].iloc[1] == -1.0

    def test_daily_freq_gap_fill(self):
        records = pd.DataFrame(
            {
                "rep": ["a", "a"],
                "d": ["2024-01-01", "2024-01-03"],
                "v": [1.0, 2.0],
            }
        )
        panel = to_panel(records, id_cols="rep", date_col="d", value_col="v", freq="D")
        assert list(panel[TIME_COL]) == list(pd.date_range("2024-01-01", periods=3, freq="D"))
        assert list(panel[TARGET_COL]) == [1.0, 0.0, 2.0]

    def test_datetimes_with_time_of_day_floor_to_period_start(self):
        records = pd.DataFrame(
            {
                "rep": ["a", "a"],
                "d": ["2024-01-15 13:45:00", "2024-01-20 09:00:00"],
                "v": [1.0, 2.0],
            }
        )
        panel = to_panel(records, id_cols="rep", date_col="d", value_col="v")
        assert list(panel[TIME_COL]) == [pd.Timestamp("2024-01-01")]
        assert list(panel[TARGET_COL]) == [3.0]

    def test_output_sorted_and_validate_panel_clean(self):
        panel = to_panel(
            _sfdc_export(), id_cols=["team", "rep"], date_col="close_date", value_col="amount"
        )
        pd.testing.assert_frame_equal(panel, validate_panel(panel))
        assert np.issubdtype(panel[TARGET_COL].dtype, np.floating)


class TestEndAnchoredFrequencies:
    """Regression: end-anchored freqs must bucket by the CONTAINING period.

    rollback-based flooring split one calendar period across two buckets
    (ME mapped 2026-03-15 to 2026-02-28 while 2026-03-31 mapped to itself);
    ``resample('ME')`` keeps all of March in one bucket labeled 2026-03-31.
    """

    def _march_deals(self):
        return pd.DataFrame(
            {
                "rep": ["a", "a", "a"],
                "close_date": ["2026-03-05", "2026-03-15", "2026-03-31"],
                "amount": [10.0, 20.0, 30.0],
            }
        )

    def test_me_march_repro_single_bucket_matching_resample(self):
        records = self._march_deals()
        panel = to_panel(
            records, id_cols="rep", date_col="close_date", value_col="amount", freq="ME"
        )
        # ONE bucket for the one calendar month, labeled at the month end
        assert list(panel[TIME_COL]) == [pd.Timestamp("2026-03-31")]
        assert list(panel[TARGET_COL]) == [60.0]
        resampled = (
            records.assign(close_date=pd.to_datetime(records["close_date"]))
            .set_index("close_date")["amount"]
            .resample("ME")
            .sum()
        )
        assert list(panel[TIME_COL]) == list(resampled.index)
        assert list(panel[TARGET_COL]) == list(resampled.to_numpy())

    def test_me_multi_month_totals_match_resample(self):
        records = pd.DataFrame(
            {
                "rep": ["a"] * 5,
                "d": ["2026-01-01", "2026-01-31", "2026-03-15", "2026-03-31", "2026-04-01"],
                "v": [1.0, 2.0, 4.0, 8.0, 16.0],
            }
        )
        panel = to_panel(records, id_cols="rep", date_col="d", value_col="v", freq="ME")
        resampled = (
            records.assign(d=pd.to_datetime(records["d"])).set_index("d")["v"].resample("ME").sum()
        )
        # Jan..Apr month ends including the empty (gap-filled) February
        assert list(panel[TIME_COL]) == list(resampled.index)
        assert list(panel[TARGET_COL]) == list(resampled.to_numpy())

    @pytest.mark.parametrize(
        "freq,expected_ds",
        [("QE", pd.Timestamp("2026-03-31")), ("YE", pd.Timestamp("2026-12-31"))],
    )
    def test_quarter_and_year_end_bucket_containing_period(self, freq, expected_ds):
        panel = to_panel(
            self._march_deals(), id_cols="rep", date_col="close_date", value_col="amount", freq=freq
        )
        assert list(panel[TIME_COL]) == [expected_ds]
        assert list(panel[TARGET_COL]) == [60.0]

    def test_month_start_behavior_unchanged(self):
        panel = to_panel(
            self._march_deals(), id_cols="rep", date_col="close_date", value_col="amount", freq="MS"
        )
        assert list(panel[TIME_COL]) == [pd.Timestamp("2026-03-01")]
        assert list(panel[TARGET_COL]) == [60.0]


class TestCountSkipnaSemantics:
    """Regression: ``agg='count'`` with a value_col must count NON-NULL values.

    It previously fell through to ``grouped.size()`` and counted NaN-valued
    rows, breaking the shared skipna semantics of sum/mean/count. Pure row
    counting remains available via ``value_col=None``.
    """

    def _with_nan_amount(self):
        records = _sfdc_export()
        records.loc[1, "amount"] = np.nan  # one of alice's two January deals
        return records

    def test_count_agg_skips_nan_values(self):
        panel = to_panel(
            self._with_nan_amount(),
            id_cols=["team", "rep"],
            date_col="close_date",
            value_col="amount",
            agg="count",
        )
        alice = panel[panel[ID_COL] == "east/alice"]
        assert list(alice[TARGET_COL]) == [1.0, 0.0, 1.0]

    def test_value_col_none_still_counts_all_rows(self):
        panel = to_panel(self._with_nan_amount(), id_cols=["team", "rep"], date_col="close_date")
        alice = panel[panel[ID_COL] == "east/alice"]
        assert list(alice[TARGET_COL]) == [2.0, 0.0, 1.0]

    def test_sum_shares_skipna_with_count(self):
        panel = to_panel(
            self._with_nan_amount(),
            id_cols=["team", "rep"],
            date_col="close_date",
            value_col="amount",
        )
        alice = panel[panel[ID_COL] == "east/alice"]
        assert list(alice[TARGET_COL]) == [100.0, 0.0, 50.0]


class TestNumericDateHints:
    """Regression: numeric date columns must never silently parse as 1970.

    ``pd.to_datetime`` reads bare ints as NANOSECOND offsets from the epoch,
    so Stripe/Mixpanel epoch-second exports and GA4 ``YYYYMMDD`` ints all
    collapsed into 1970-01-01 buckets. ``date_unit`` / ``date_format`` are
    the explicit hints; with neither, integer/float date columns now raise.
    """

    @staticmethod
    def _epoch(day: str) -> int:
        # naive pd.Timestamp.timestamp() treats the wall time as UTC, matching
        # the naive datetimes produced by pd.to_datetime(..., unit=...)
        return int(pd.Timestamp(day).timestamp())

    def _stripe_like(self, scale: int = 1) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "rep": ["a", "a", "a"],
                "created": [
                    self._epoch("2026-01-15") * scale,
                    self._epoch("2026-02-10") * scale,
                    self._epoch("2026-03-05") * scale,
                ],
                "amount": [10.0, 20.0, 30.0],
            }
        )

    def test_epoch_seconds_bucket_into_2026_months(self):
        panel = to_panel(
            self._stripe_like(),
            id_cols="rep",
            date_col="created",
            value_col="amount",
            date_unit="s",
        )
        assert list(panel[TIME_COL]) == [
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-02-01"),
            pd.Timestamp("2026-03-01"),
        ]
        assert list(panel[TARGET_COL]) == [10.0, 20.0, 30.0]

    def test_epoch_milliseconds(self):
        panel = to_panel(
            self._stripe_like(scale=1000),
            id_cols="rep",
            date_col="created",
            value_col="amount",
            date_unit="ms",
        )
        assert list(panel[TIME_COL]) == [
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-02-01"),
            pd.Timestamp("2026-03-01"),
        ]

    def test_epoch_seconds_as_digit_strings_still_parse(self):
        # a CSV read with dtype=str delivers epoch seconds as digit strings
        records = self._stripe_like()
        records["created"] = records["created"].astype(str)
        panel = to_panel(
            records, id_cols="rep", date_col="created", value_col="amount", date_unit="s"
        )
        assert panel[TIME_COL].min() == pd.Timestamp("2026-01-01")
        assert panel[TIME_COL].max() == pd.Timestamp("2026-03-01")

    @pytest.mark.parametrize("cast", [int, float, str])
    def test_date_format_yyyymmdd_across_dtypes(self, cast):
        # GA4 event_date arrives as int (read_csv), float (numeric-text
        # cleaning), or str (dtype=str) — all must land on 2026-01-15
        records = pd.DataFrame(
            {
                "rep": ["a", "a"],
                "event_date": [cast(20260115), cast(20260116)],
                "v": [1.0, 2.0],
            }
        )
        panel = to_panel(
            records,
            id_cols="rep",
            date_col="event_date",
            value_col="v",
            freq="D",
            date_format="%Y%m%d",
        )
        assert list(panel[TIME_COL]) == [
            pd.Timestamp("2026-01-15"),
            pd.Timestamp("2026-01-16"),
        ]
        assert list(panel[TARGET_COL]) == [1.0, 2.0]

    def test_bare_integer_dates_without_hint_raise(self):
        records = self._stripe_like()
        with pytest.raises(DataContractError, match="date_unit"):
            to_panel(records, id_cols="rep", date_col="created", value_col="amount")

    def test_bare_float_dates_without_hint_raise(self):
        records = self._stripe_like()
        records["created"] = records["created"].astype(float)
        with pytest.raises(DataContractError, match="date_format"):
            to_panel(records, id_cols="rep", date_col="created", value_col="amount")

    def test_string_dates_unaffected_by_guard(self):
        panel = to_panel(_sfdc_export(), id_cols="rep", date_col="close_date")
        assert panel[TIME_COL].min() == pd.Timestamp("2024-01-01")

    def test_both_hints_raise(self):
        with pytest.raises(ValueError, match="not both"):
            to_panel(
                self._stripe_like(),
                id_cols="rep",
                date_col="created",
                value_col="amount",
                date_unit="s",
                date_format="%Y%m%d",
            )

    def test_unknown_date_unit_raises(self):
        with pytest.raises(ValueError, match="date_unit"):
            to_panel(
                self._stripe_like(),
                id_cols="rep",
                date_col="created",
                value_col="amount",
                date_unit="fortnights",
            )

    def test_date_unit_on_unconvertible_strings_raises_contract_error(self):
        records = self._stripe_like()
        records["created"] = ["2026-01-15", "2026-02-10", "2026-03-05"]
        with pytest.raises(DataContractError, match="unparseable"):
            to_panel(
                records, id_cols="rep", date_col="created", value_col="amount", date_unit="s"
            )


class TestNaNFillValue:
    """Regression: fill_value=NaN must not trip validate_panel's NaN check.

    to_panel now validates with ``allow_missing=True`` when the fill value
    is NaN — the gaps are deliberately marked missing for downstream
    imputation instead of being invented as zeros.
    """

    def _ragged(self):
        return pd.DataFrame(
            {
                "rep": ["a"] * 2 + ["b"] * 2,
                "d": ["2026-01-05", "2026-03-10", "2026-03-02", "2026-05-20"],
                "amt": [1.0, 2.0, 3.0, 4.0],
            }
        )

    def test_span_panel_nan_fill_round_trips(self):
        panel = to_panel(
            self._ragged(),
            id_cols=["rep"],
            date_col="d",
            value_col="amt",
            span="panel",
            fill_value=np.nan,
        )
        # b had not started in January: marked missing, not invented
        b_jan = panel[(panel[ID_COL] == "b") & (panel[TIME_COL] == "2026-01-01")]
        assert b_jan[TARGET_COL].isna().all() and len(b_jan) == 1
        # round-trips through the contract with allow_missing
        pd.testing.assert_frame_equal(panel, validate_panel(panel, allow_missing=True))

    def test_series_span_nan_fill_for_interior_gaps(self):
        panel = to_panel(
            self._ragged(), id_cols=["rep"], date_col="d", value_col="amt", fill_value=np.nan
        )
        a = panel[panel[ID_COL] == "a"]
        assert a[TARGET_COL].isna().tolist() == [False, True, False]

    def test_default_zero_fill_still_validates_strictly(self):
        panel = to_panel(self._ragged(), id_cols=["rep"], date_col="d", value_col="amt")
        assert not panel[TARGET_COL].isna().any()
        pd.testing.assert_frame_equal(panel, validate_panel(panel))


class TestNullKeysRejected:
    """Regression: null date/id keys were SILENTLY DELETED from the panel.

    ``work.groupby([unique_id, ds])`` uses pandas' default ``dropna=True``,
    so a blank/NaT close date (the single most common CRM null: an open
    deal) or a null id cell removed those rows from the output with no
    warning — 4 rows totalling 100.0 in, 3 rows totalling 70.0 out. The id
    path was worse than a drop in principle: ``.astype(str)`` on pandas 2+
    keeps NA (so the row vanished the same way) and on any object column
    that stringified it would have invented a bogus ``"nan"`` series.

    A key is not a value: a row that cannot say WHICH series it belongs to
    or WHEN it happened cannot be bucketed at all, so it is rejected loudly,
    matching ``validate_panel``'s treatment of null ``unique_id``/``ds``.
    Callers who genuinely want those rows gone must drop them explicitly.
    """

    def _open_deal(self, close):
        """Three closed deals plus one open deal whose close date is ``close``."""
        return pd.DataFrame(
            {
                "rep": ["ana", "ana", "ana", "ana"],
                "close": ["2026-01-05", "2026-02-10", "2026-03-04", close],
                "amount": [10.0, 20.0, 40.0, 30.0],
            }
        )

    @pytest.mark.parametrize("blank", ["", None, pd.NaT, np.nan])
    def test_null_or_blank_date_raises_instead_of_deleting_revenue(self, blank):
        with pytest.raises(DataContractError, match=r"'close'.*1 row\(s\).*null"):
            to_panel(self._open_deal(blank), id_cols="rep", date_col="close", value_col="amount")

    def test_error_names_the_offending_rows(self):
        records = self._open_deal(None)
        with pytest.raises(DataContractError, match="3"):  # the offending row label
            to_panel(records, id_cols="rep", date_col="close", value_col="amount")

    def test_no_revenue_is_lost_once_the_null_row_is_dropped_explicitly(self):
        records = self._open_deal(None)
        kept = records.dropna(subset=["close"])
        panel = to_panel(kept, id_cols="rep", date_col="close", value_col="amount")
        assert panel[TARGET_COL].sum() == kept["amount"].sum() == 70.0

    def test_clean_records_keep_every_dollar(self):
        records = self._open_deal("2026-04-01")
        panel = to_panel(records, id_cols="rep", date_col="close", value_col="amount")
        assert panel[TARGET_COL].sum() == records["amount"].sum() == 100.0

    def test_epoch_dates_with_a_null_raise_too(self):
        # the date_unit path parses NaN -> NaT just as the string path does
        records = pd.DataFrame(
            {
                "rep": ["a", "a"],
                "created": [float(pd.Timestamp("2026-01-15").timestamp()), np.nan],
                "amount": [10.0, 90.0],
            }
        )
        with pytest.raises(DataContractError, match="null"):
            to_panel(records, id_cols="rep", date_col="created", value_col="amount", date_unit="s")

    @pytest.mark.parametrize("null", [None, np.nan, pd.NA])
    def test_null_id_raises_instead_of_dropping_the_row(self, null):
        records = pd.DataFrame(
            {
                "rep": ["ana", null],
                "close": ["2026-01-05", "2026-01-09"],
                "amount": [100.0, 50.0],
            }
        )
        with pytest.raises(DataContractError, match=r"'rep'.*1 row\(s\).*null"):
            to_panel(records, id_cols="rep", date_col="close", value_col="amount")

    def test_null_in_any_id_level_raises(self):
        records = _sfdc_export()
        records.loc[2, "team"] = None
        with pytest.raises(DataContractError, match="'team'"):
            to_panel(
                records, id_cols=["team", "rep"], date_col="close_date", value_col="amount"
            )

    def test_null_id_is_rejected_before_it_can_be_stringified(self):
        """A null id must never reach ``.astype(str)`` and become ``"west/nan"``."""
        records = _sfdc_export()
        records["rep"] = records["rep"].astype(object)
        records.loc[3, "rep"] = None
        with pytest.raises(DataContractError, match="'rep'"):
            to_panel(
                records, id_cols=["team", "rep"], date_col="close_date", value_col="amount"
            )


class TestToPanelValidation:
    def test_missing_id_col_raises(self):
        with pytest.raises(DataContractError, match="missing"):
            to_panel(_sfdc_export(), id_cols=["team", "nope"], date_col="close_date")

    def test_missing_date_col_raises(self):
        with pytest.raises(DataContractError, match="missing"):
            to_panel(_sfdc_export(), id_cols="rep", date_col="nope")

    def test_missing_value_col_raises(self):
        with pytest.raises(DataContractError, match="missing"):
            to_panel(_sfdc_export(), id_cols="rep", date_col="close_date", value_col="nope")

    @pytest.mark.filterwarnings("ignore::UserWarning")  # pandas warns pre-raise
    def test_unparseable_dates_raise(self):
        records = _sfdc_export()
        records.loc[0, "close_date"] = "not-a-date"
        with pytest.raises(DataContractError, match="date"):
            to_panel(records, id_cols="rep", date_col="close_date")

    def test_unknown_agg_raises(self):
        with pytest.raises(ValueError, match="agg"):
            to_panel(
                _sfdc_export(),
                id_cols="rep",
                date_col="close_date",
                value_col="amount",
                agg="median",
            )

    def test_empty_records_raise(self):
        empty = _sfdc_export().iloc[0:0]
        with pytest.raises(DataContractError, match="empty"):
            to_panel(empty, id_cols="rep", date_col="close_date")

    def test_non_dataframe_raises(self):
        with pytest.raises(DataContractError, match="DataFrame"):
            to_panel([1, 2, 3], id_cols="rep", date_col="close_date")


def test_span_panel_aligns_ragged_series_for_hierarchies():
    """span='panel' aligns every series to the global range (0-filled)."""
    recs = pd.DataFrame(
        {
            "rep": ["a"] * 2 + ["b"] * 2,
            "d": ["2026-01-05", "2026-03-10", "2026-03-02", "2026-05-20"],
            "amt": [1.0, 2.0, 3.0, 4.0],
        }
    )
    default = to_panel(recs, id_cols=["rep"], date_col="d", value_col="amt")
    assert default.groupby("unique_id")["ds"].min().nunique() == 2  # ragged

    aligned = to_panel(recs, id_cols=["rep"], date_col="d", value_col="amt", span="panel")
    spans = aligned.groupby("unique_id")["ds"].agg(["min", "max", "size"])
    assert spans["min"].nunique() == 1 and spans["max"].nunique() == 1
    assert (spans["size"] == 5).all()  # Jan..May for both series
    assert aligned.loc[
        (aligned["unique_id"] == "b") & (aligned["ds"] == "2026-01-01"), "y"
    ].item() == 0.0

    with pytest.raises(ValueError, match="span"):
        to_panel(recs, id_cols=["rep"], date_col="d", value_col="amt", span="global")
