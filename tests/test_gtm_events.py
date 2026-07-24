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
