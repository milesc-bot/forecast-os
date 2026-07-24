"""Tests for currency normalization (preprocessing.currency).

Deal/opportunity frames are deal-grain (one row per opportunity), so these
classes are standalone helpers, not (unique_id, ds, y) panel transforms. The
correctness landmine they defuse: to_panel sums amounts blindly, so mixing
EUR and USD rows produces a meaningless total unless amounts are normalized
to one reporting currency first.
"""

import numpy as np
import pandas as pd
import pytest

from forecast_os.core.exceptions import DataContractError, ForecastOSError
from forecast_os.gtm import to_panel
from forecast_os.preprocessing.currency import (
    CurrencyNormalizer,
    convert_currency,
    guard_single_currency,
)


def _deals():
    """Three closed deals in three currencies (one already in USD)."""
    return pd.DataFrame(
        {
            "opp_id": [1, 2, 3],
            "amount": [100.0, 200.0, 50.0],
            "currency": ["EUR", "GBP", "USD"],
            "stage": ["won", "won", "won"],
        }
    )


def _dated_rates():
    """A dated rate table where the EUR->USD rate steps up mid-year."""
    return pd.DataFrame(
        {
            "currency": ["EUR", "EUR", "GBP"],
            "date": ["2024-01-01", "2024-06-01", "2024-01-01"],
            "rate": [1.05, 1.10, 1.27],
        }
    )


class TestConvertCurrencyDict:
    def test_scalar_dict_rates_hand_checked(self):
        out = convert_currency(_deals(), {"EUR": 1.08, "GBP": 1.27}, to="USD")
        # 100*1.08=108, 200*1.27=254, 50 already USD (rate 1.0)
        assert out["amount"].tolist() == [108.0, 254.0, 50.0]

    def test_tuple_dict_rates_hand_checked(self):
        rates = {("EUR", "USD"): 1.08, ("GBP", "USD"): 1.27}
        out = convert_currency(_deals(), rates, to="USD")
        assert out["amount"].tolist() == [108.0, 254.0, 50.0]

    def test_passthrough_row_already_in_to(self):
        # The USD row must be untouched regardless of what rates contains.
        out = convert_currency(_deals(), {"EUR": 1.08, "GBP": 1.27}, to="USD")
        assert out.loc[out["currency"] == "USD", "amount"].item() == 50.0

    def test_convert_to_non_usd_reporting_currency(self):
        deals = pd.DataFrame({"amount": [100.0, 100.0], "currency": ["EUR", "USD"]})
        out = convert_currency(deals, {"USD": 0.9}, to="EUR")
        # EUR passthrough, USD*0.9
        assert out["amount"].tolist() == [100.0, 90.0]

    def test_missing_rate_raises_naming_currency(self):
        with pytest.raises(DataContractError) as exc:
            convert_currency(_deals(), {"EUR": 1.08}, to="USD")
        assert "GBP" in str(exc.value)

    def test_does_not_overwrite_currency_column(self):
        # convert_currency only rewrites the amount column; the currency label
        # is CurrencyNormalizer's job.
        out = convert_currency(_deals(), {"EUR": 1.08, "GBP": 1.27}, to="USD")
        assert out["currency"].tolist() == ["EUR", "GBP", "USD"]

    def test_does_not_mutate_input(self):
        deals = _deals()
        convert_currency(deals, {"EUR": 1.08, "GBP": 1.27}, to="USD")
        assert deals["amount"].tolist() == [100.0, 200.0, 50.0]

    def test_mixed_tuple_and_scalar_keys_raises(self):
        with pytest.raises(ForecastOSError):
            convert_currency(_deals(), {("EUR", "USD"): 1.08, "GBP": 1.27}, to="USD")

    def test_non_dataframe_raises(self):
        with pytest.raises(DataContractError):
            convert_currency([1, 2, 3], {"EUR": 1.08}, to="USD")

    def test_missing_amount_column_raises(self):
        deals = pd.DataFrame({"currency": ["EUR"]})
        with pytest.raises(DataContractError):
            convert_currency(deals, {"EUR": 1.08}, to="USD")

    def test_missing_currency_column_raises(self):
        deals = pd.DataFrame({"amount": [100.0]})
        with pytest.raises(DataContractError):
            convert_currency(deals, {"EUR": 1.08}, to="USD")

    def test_all_rows_in_to_needs_no_rates(self):
        deals = pd.DataFrame({"amount": [1.0, 2.0], "currency": ["USD", "USD"]})
        out = convert_currency(deals, {}, to="USD")
        assert out["amount"].tolist() == [1.0, 2.0]


class TestConvertCurrencyDated:
    def test_asof_join_picks_prior_rate(self):
        deals = pd.DataFrame(
            {
                "opp_id": [1, 2, 3, 4],
                "amount": [100.0, 100.0, 200.0, 50.0],
                "currency": ["EUR", "EUR", "GBP", "USD"],
                "close_date": ["2024-03-15", "2024-07-01", "2024-02-10", "2024-07-01"],
            }
        )
        out = convert_currency(
            deals, _dated_rates(), to="USD", as_of_col="date", date_col="close_date"
        )
        # EUR 2024-03-15 -> latest rate on/before is 2024-01-01 (1.05) -> 105
        # EUR 2024-07-01 -> latest is 2024-06-01 (1.10) -> 110
        # GBP 2024-02-10 -> 2024-01-01 (1.27) -> 254
        # USD -> passthrough 50
        assert out["amount"].tolist() == pytest.approx([105.0, 110.0, 254.0, 50.0])

    def test_asof_preserves_row_order(self):
        # Rows deliberately out of date order; output must stay row-aligned.
        deals = pd.DataFrame(
            {
                "amount": [100.0, 100.0],
                "currency": ["EUR", "EUR"],
                "close_date": ["2024-07-01", "2024-03-15"],
            }
        )
        out = convert_currency(
            deals, _dated_rates(), to="USD", as_of_col="date", date_col="close_date"
        )
        assert out["amount"].tolist() == pytest.approx([110.0, 105.0])

    def test_asof_missing_prior_rate_raises_naming_currency_and_date(self):
        deals = pd.DataFrame(
            {
                "amount": [100.0],
                "currency": ["EUR"],
                "close_date": ["2023-12-01"],
            }
        )
        with pytest.raises(DataContractError) as exc:
            convert_currency(
                deals, _dated_rates(), to="USD", as_of_col="date", date_col="close_date"
            )
        assert "EUR" in str(exc.value)
        assert "2023-12-01" in str(exc.value)

    def test_dated_requires_both_cols(self):
        with pytest.raises(ValueError):
            convert_currency(_deals(), _dated_rates(), to="USD", as_of_col="date")

    def test_dict_rates_with_dated_cols_raises(self):
        deals = pd.DataFrame({"amount": [1.0], "currency": ["EUR"], "close_date": ["2024-01-01"]})
        with pytest.raises(ValueError):
            convert_currency(
                deals, {"EUR": 1.08}, to="USD", as_of_col="date", date_col="close_date"
            )

    def test_dataframe_rates_without_cols_raises(self):
        with pytest.raises(ValueError):
            convert_currency(_deals(), _dated_rates(), to="USD")

    def test_missing_rate_column_in_dated_frame_raises(self):
        bad = _dated_rates().drop(columns=["rate"])
        deals = pd.DataFrame({"amount": [1.0], "currency": ["EUR"], "close_date": ["2024-02-01"]})
        with pytest.raises(DataContractError):
            convert_currency(deals, bad, to="USD", as_of_col="date", date_col="close_date")


class TestGuardSingleCurrency:
    def test_mixed_raises_naming_currencies(self):
        with pytest.raises(DataContractError) as exc:
            guard_single_currency(_deals())
        msg = str(exc.value)
        assert "EUR" in msg and "GBP" in msg

    def test_single_currency_passes(self):
        df = pd.DataFrame({"amount": [1.0, 2.0], "currency": ["USD", "USD"]})
        assert guard_single_currency(df) is None

    def test_single_with_nulls_passes(self):
        df = pd.DataFrame({"amount": [1.0, 2.0], "currency": ["EUR", None]})
        assert guard_single_currency(df) is None

    def test_all_null_passes(self):
        df = pd.DataFrame({"amount": [1.0], "currency": [None]})
        assert guard_single_currency(df) is None

    def test_missing_column_raises(self):
        with pytest.raises(DataContractError):
            guard_single_currency(pd.DataFrame({"amount": [1.0]}))

    def test_non_dataframe_raises(self):
        with pytest.raises(DataContractError):
            guard_single_currency("not a frame")

    def test_custom_currency_col(self):
        df = pd.DataFrame({"ccy": ["EUR", "GBP"]})
        with pytest.raises(DataContractError):
            guard_single_currency(df, currency_col="ccy")


class TestCurrencyNormalizer:
    def test_constructor_args_stored_as_attributes(self):
        rates = {"EUR": 1.08}
        norm = CurrencyNormalizer(rates, currency_col="ccy", to="GBP", amount_col="amt")
        assert norm.rates is rates
        assert norm.currency_col == "ccy"
        assert norm.to == "GBP"
        assert norm.amount_col == "amt"

    def test_fit_returns_self(self):
        norm = CurrencyNormalizer({"EUR": 1.08})
        assert norm.fit(_deals()) is norm

    def test_transform_converts_and_sets_currency(self):
        norm = CurrencyNormalizer({"EUR": 1.08, "GBP": 1.27}, to="USD")
        out = norm.fit_transform(_deals())
        assert out["amount"].tolist() == [108.0, 254.0, 50.0]
        # Unlike convert_currency, the normalizer relabels the currency column.
        assert out["currency"].tolist() == ["USD", "USD", "USD"]
        # And the relabeled frame passes the mixed-currency guard.
        assert guard_single_currency(out) is None

    def test_pipeline_like_roundtrip_and_aggregates(self):
        deals = pd.DataFrame(
            {
                "rep": ["alice", "alice", "bob"],
                "close_date": ["2024-01-10", "2024-01-20", "2024-01-15"],
                "amount": [100.0, 200.0, 50.0],
                "currency": ["EUR", "EUR", "GBP"],
            }
        )

        class _MiniPipeline:
            def __init__(self, steps):
                self.steps = steps

            def fit_transform(self, df):
                for _, step in self.steps:
                    df = step.fit(df).transform(df)
                return df

        pipe = _MiniPipeline([("fx", CurrencyNormalizer({"EUR": 1.08, "GBP": 1.27}, to="USD"))])
        converted = pipe.fit_transform(deals)
        guard_single_currency(converted)  # normalized -> no raise
        # hand-checked USD: 108, 216, 63.5
        assert converted["amount"].tolist() == [108.0, 216.0, 63.5]

        panel = to_panel(converted, id_cols="rep", date_col="close_date", value_col="amount")
        alice = panel[panel["unique_id"] == "alice"]
        bob = panel[panel["unique_id"] == "bob"]
        # alice Jan total 108+216=324 (now a meaningful single-currency sum)
        assert alice["y"].tolist() == [324.0]
        assert bob["y"].tolist() == [63.5]

    def test_normalizer_missing_rate_raises(self):
        norm = CurrencyNormalizer({"EUR": 1.08}, to="USD")
        with pytest.raises(DataContractError):
            norm.fit_transform(_deals())  # GBP has no rate


def test_no_new_runtime_deps():
    # Sanity: the module leans only on numpy/pandas (core stack).
    import forecast_os.preprocessing.currency as mod

    assert mod.np is np
    assert mod.pd is pd
