"""Tests for the GTM funnel utilities (gtm.funnel)."""

import numpy as np
import pandas as pd
import pytest

from forecast_os.core.exceptions import ForecastOSError
from forecast_os.core.types import ID_COL, TARGET_COL, TIME_COL
from forecast_os.gtm import conversion_rates, propagate, stage_panel


def _crm_records():
    """One row per funnel event: Jan has 4 leads / 2 mqls, Feb 2 leads, Mar 1 mql."""
    rows = (
        [("lead", "2024-01-05")] * 4
        + [("mql", "2024-01-20")] * 2
        + [("lead", "2024-02-10")] * 2
        + [("mql", "2024-03-15")]
    )
    return pd.DataFrame(rows, columns=["stage", "created"])


def _stage_counts():
    """Hand-built stage panel: lead [10, 0, 4], mql [5, 0, 2] over Jan-Mar."""
    ds = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])
    return pd.DataFrame(
        {
            ID_COL: ["lead"] * 3 + ["mql"] * 3,
            TIME_COL: list(ds) * 2,
            TARGET_COL: [10.0, 0.0, 4.0, 5.0, 0.0, 2.0],
        }
    )


class TestStagePanel:
    def test_counts_per_stage(self):
        panel = stage_panel(_crm_records(), stage_col="stage", date_col="created")
        assert set(panel[ID_COL]) == {"lead", "mql"}
        lead = panel[panel[ID_COL] == "lead"]
        assert list(lead[TARGET_COL]) == [4.0, 2.0]  # Jan, Feb (own range only)
        mql = panel[panel[ID_COL] == "mql"]
        assert list(mql[TARGET_COL]) == [2.0, 0.0, 1.0]  # Jan, Feb (gap), Mar

    def test_id_cols_prefix(self):
        records = _crm_records()
        records["team"] = "east"
        panel = stage_panel(
            records, stage_col="stage", date_col="created", id_cols=["team"]
        )
        assert set(panel[ID_COL]) == {"east/lead", "east/mql"}


class TestConversionRates:
    def test_hand_computed_rates_including_zero_over_zero(self):
        rates = conversion_rates(_stage_counts(), stages=["lead", "mql"])
        assert set(rates[ID_COL]) == {"lead->mql"}
        y = rates.sort_values(TIME_COL)[TARGET_COL].to_numpy()
        # Jan 5/10, Feb 0/0 -> NaN, Mar 2/4
        assert y[0] == 0.5
        assert np.isnan(y[1])
        assert y[2] == 0.5

    def test_three_stages_two_rate_series(self):
        df = _stage_counts()
        extra = pd.DataFrame(
            {
                ID_COL: ["sql"] * 3,
                TIME_COL: pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
                TARGET_COL: [1.0, 0.0, 1.0],
            }
        )
        rates = conversion_rates(pd.concat([df, extra]), stages=["lead", "mql", "sql"])
        assert set(rates[ID_COL]) == {"lead->mql", "mql->sql"}
        mql_sql = rates[rates[ID_COL] == "mql->sql"].sort_values(TIME_COL)
        assert mql_sql[TARGET_COL].iloc[0] == 0.2  # 1/5
        assert np.isnan(mql_sql[TARGET_COL].iloc[1])  # 0/0
        assert mql_sql[TARGET_COL].iloc[2] == 0.5  # 1/2

    def test_missing_next_stage_period_counts_as_zero(self):
        # lead observed Jan-Feb, mql only Jan: Feb mql treated as 0
        df = pd.DataFrame(
            {
                ID_COL: ["lead", "lead", "mql"],
                TIME_COL: pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01"]),
                TARGET_COL: [10.0, 5.0, 4.0],
            }
        )
        rates = conversion_rates(df, stages=["lead", "mql"]).sort_values(TIME_COL)
        assert list(rates[TARGET_COL]) == [0.4, 0.0]

    def test_prefixed_ids_are_preserved(self):
        df = _stage_counts()
        df[ID_COL] = "east/" + df[ID_COL]
        rates = conversion_rates(df, stages=["lead", "mql"])
        assert set(rates[ID_COL]) == {"east/lead->mql"}
        y = rates.sort_values(TIME_COL)[TARGET_COL].to_numpy()
        assert y[0] == 0.5 and np.isnan(y[1]) and y[2] == 0.5

    def test_unknown_stage_raises(self):
        with pytest.raises(ForecastOSError, match="sal"):
            conversion_rates(_stage_counts(), stages=["lead", "sal"])

    def test_fewer_than_two_stages_raises(self):
        with pytest.raises(ValueError, match="two"):
            conversion_rates(_stage_counts(), stages=["lead"])


class TestPropagate:
    def _top(self):
        return pd.DataFrame(
            {
                ID_COL: ["pipe", "pipe"],
                TIME_COL: pd.to_datetime(["2024-04-01", "2024-05-01"]),
                "yhat": [100.0, 200.0],
            }
        )

    def test_chained_volumes(self):
        out = propagate(self._top(), {"mql": 0.5, "sql": 0.4})
        assert list(out["mql"]) == [50.0, 100.0]
        assert list(out["sql"]) == [20.0, 40.0]  # 100 * 0.5 * 0.4, 200 * 0.5 * 0.4
        # original forecast columns preserved
        assert list(out["yhat"]) == [100.0, 200.0]

    def test_accepts_y_column_panel(self):
        top = self._top().rename(columns={"yhat": "y"})
        out = propagate(top, {"won": 0.25})
        assert list(out["won"]) == [25.0, 50.0]

    def test_does_not_mutate_input(self):
        top = self._top()
        propagate(top, {"mql": 0.5})
        assert "mql" not in top.columns

    def test_rate_outside_unit_interval_raises(self):
        with pytest.raises(ValueError, match="rate"):
            propagate(self._top(), {"mql": 1.5})
        with pytest.raises(ValueError, match="rate"):
            propagate(self._top(), {"mql": -0.1})

    def test_missing_forecast_column_raises(self):
        bad = self._top().drop(columns=["yhat"])
        with pytest.raises(ForecastOSError, match="yhat"):
            propagate(bad, {"mql": 0.5})

    def test_empty_rates_raises(self):
        with pytest.raises(ValueError, match="rates"):
            propagate(self._top(), {})


class TestConversionRatesEmptyAndSeparatorInStageName:
    """Regressions for the bare pandas concat crash and `sep`-mangled stage names
    (v0.9.0 audit SHOULD-2).

    ``conversion_rates`` checks stage presence panel-WIDE but emits rate series
    per prefix group, so a pair whose two stages never share a prefix left
    ``frames`` empty and pandas raised "No objects to concatenate" with no domain
    context. Separately, ids were split with a blind ``rpartition(sep)``, which
    mangles a stage name that itself contains ``sep`` — Salesforce's stock
    ``"Negotiation/Review"`` picklist value became prefix "Negotiation" / stage
    "Review", so the real stage name was reported missing and the mangled one hit
    the concat crash.
    """

    def test_disjoint_prefix_groups_raise_a_domain_error(self):
        df = pd.DataFrame(
            {
                ID_COL: ["east/lead", "west/mql"],
                TIME_COL: pd.to_datetime(["2024-01-01", "2024-01-01"]),
                TARGET_COL: [10.0, 5.0],
            }
        )
        with pytest.raises(ForecastOSError, match="prefix"):
            conversion_rates(df, stages=["lead", "mql"])

    def test_stage_name_containing_sep_round_trips_through_stage_panel(self):
        records = pd.DataFrame(
            {
                "stage": ["Qualification"] * 4 + ["Negotiation/Review"] * 2,
                "created": ["2024-01-05"] * 4 + ["2024-01-20"] * 2,
            }
        )
        panel = stage_panel(records, stage_col="stage", date_col="created")
        rates = conversion_rates(panel, stages=["Qualification", "Negotiation/Review"])
        assert set(rates[ID_COL]) == {"Qualification->Negotiation/Review"}
        assert rates[TARGET_COL].iloc[0] == pytest.approx(0.5)

    def test_prefix_plus_stage_name_containing_sep(self):
        ds = pd.to_datetime(["2024-01-01"])
        df = pd.DataFrame(
            {
                ID_COL: ["east/Qualification", "east/Negotiation/Review"],
                TIME_COL: list(ds) * 2,
                TARGET_COL: [8.0, 2.0],
            }
        )
        rates = conversion_rates(df, stages=["Qualification", "Negotiation/Review"])
        assert set(rates[ID_COL]) == {"east/Qualification->Negotiation/Review"}
        assert rates[TARGET_COL].iloc[0] == pytest.approx(0.25)

    def test_unknown_stage_message_is_not_confused_by_a_sep_in_the_name(self):
        ds = pd.to_datetime(["2024-01-01"])
        df = pd.DataFrame(
            {
                ID_COL: ["Qualification", "Negotiation/Review"],
                TIME_COL: list(ds) * 2,
                TARGET_COL: [8.0, 2.0],
            }
        )
        # 'Review' alone is not a stage in this panel; the error must say so
        # rather than the concat crash the mangled split used to produce.
        with pytest.raises(ForecastOSError, match="Review"):
            conversion_rates(df, stages=["Qualification", "Review"])
