"""Tests for deal-level win-probability scoring and the probabilistic
pipeline forecast (gtm.opportunities)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit

from forecast_os.core.exceptions import DataContractError, ForecastOSError, NotFittedError
from forecast_os.gtm.opportunities import DealScorer, weighted_pipeline

Z80 = float(stats.norm.ppf(0.9))


def _log_loss(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def make_deals(n=6000, seed=0, closed=True):
    """Synthetic deals with a KNOWN logit: won ~ Bernoulli(sigmoid(2 x1 - x2 + 0.5))."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logit = 2.0 * x1 - 1.0 * x2 + 0.5
    p = expit(logit)
    won = rng.random(n) < p
    amount = rng.uniform(1_000.0, 50_000.0, size=n)
    region = rng.choice(["EMEA", "AMER"], size=n)
    frame = pd.DataFrame(
        {
            "opp_id": np.arange(n),
            "amount": amount,
            "stage": rng.choice(["prospect", "negotiation"], size=n),
            "region": region,
            "x1": x1,
            "x2": x2,
        }
    )
    if closed:
        frame["won"] = won
    return frame


class TestDealScorerFit:
    def test_recovers_coefficient_signs(self):
        scorer = DealScorer(seed=0).fit(make_deals(n=6000, seed=1))
        assert isinstance(scorer.coef_, pd.Series)
        assert list(scorer.coef_.index) == ["x1", "x2"]
        assert scorer.coef_["x1"] > 0  # true coefficient +2
        assert scorer.coef_["x2"] < 0  # true coefficient -1
        # x1 has the larger absolute effect
        assert abs(scorer.coef_["x1"]) > abs(scorer.coef_["x2"])

    def test_explicit_features_argument_selects_columns(self):
        deals = make_deals(n=2000, seed=2)
        scorer = DealScorer().fit(deals, features=["x1"])
        assert list(scorer.coef_.index) == ["x1"]

    def test_beats_constant_baseline_on_holdout_logloss(self):
        train = make_deals(n=6000, seed=1)
        test = make_deals(n=6000, seed=2)
        scorer = DealScorer(seed=0).fit(train)
        p_model = scorer.predict_proba(test)
        base_rate = float(train["won"].mean())
        model_ll = _log_loss(test["won"].to_numpy(), p_model.to_numpy())
        base_ll = _log_loss(test["won"].to_numpy(), np.full(len(test), base_rate))
        assert model_ll < base_ll - 0.02

    def test_predicted_probabilities_are_calibrated(self):
        train = make_deals(n=8000, seed=1)
        test = make_deals(n=8000, seed=3)
        scorer = DealScorer(calibrate=True, seed=0).fit(train)
        p = scorer.predict_proba(test).to_numpy()
        empirical = float(test["won"].mean())
        assert abs(p.mean() - empirical) < 0.03

    def test_predict_proba_strictly_between_zero_and_one_and_aligned(self):
        train = make_deals(n=2000, seed=1)
        open_deals = make_deals(n=500, seed=9, closed=False)
        open_deals = open_deals.set_index(open_deals["opp_id"] * 7 + 3)  # non-default index
        scorer = DealScorer(seed=0).fit(train)
        p = scorer.predict_proba(open_deals)
        assert isinstance(p, pd.Series)
        assert len(p) == len(open_deals)
        assert list(p.index) == list(open_deals.index)
        assert (p > 0).all() and (p < 1).all()

    def test_float_and_bool_targets_both_accepted(self):
        deals = make_deals(n=1500, seed=4)
        s_bool = DealScorer(calibrate=False, seed=0).fit(deals)
        deals_float = deals.copy()
        deals_float["won"] = deals_float["won"].astype(float)
        s_float = DealScorer(calibrate=False, seed=0).fit(deals_float)
        np.testing.assert_allclose(s_bool.coef_.to_numpy(), s_float.coef_.to_numpy())

    def test_constructor_args_stored_as_attributes(self):
        scorer = DealScorer(features=["x1", "x2"], l2=2.5, calibrate=False, seed=7)
        assert scorer.features == ["x1", "x2"]
        assert scorer.l2 == 2.5
        assert scorer.calibrate is False
        assert scorer.seed == 7

    def test_predict_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            DealScorer().predict_proba(make_deals(n=10, seed=0, closed=False))


class TestDealScorerValidation:
    def test_non_binary_target_raises(self):
        deals = make_deals(n=200, seed=1)
        deals["won"] = np.arange(len(deals)) % 3  # values {0, 1, 2}
        with pytest.raises(DataContractError, match="target"):
            DealScorer().fit(deals)

    def test_string_target_raises(self):
        deals = make_deals(n=200, seed=1)
        deals["won"] = deals["won"].map({True: "yes", False: "no"})
        with pytest.raises(DataContractError):
            DealScorer().fit(deals)

    def test_missing_target_column_raises(self):
        deals = make_deals(n=200, seed=1, closed=False)
        with pytest.raises(DataContractError, match="won"):
            DealScorer().fit(deals)

    def test_no_numeric_features_raises(self):
        deals = make_deals(n=200, seed=1)[["opp_id", "amount", "stage", "region", "won"]]
        with pytest.raises(DataContractError, match="numeric"):
            DealScorer().fit(deals)

    def test_explicit_non_numeric_feature_raises(self):
        deals = make_deals(n=200, seed=1)
        with pytest.raises(DataContractError):
            DealScorer().fit(deals, features=["stage"])

    def test_nan_feature_raises(self):
        deals = make_deals(n=200, seed=1)
        deals.loc[0, "x1"] = np.nan
        with pytest.raises(DataContractError):
            DealScorer().fit(deals, features=["x1", "x2"])

    def test_single_class_target_raises(self):
        deals = make_deals(n=200, seed=1)
        deals["won"] = True
        with pytest.raises(DataContractError):
            DealScorer().fit(deals)


class TestWeightedPipeline:
    def test_expected_equals_hand_computed_sum(self):
        rng = np.random.default_rng(0)
        n = 40
        open_deals = pd.DataFrame(
            {"opp_id": np.arange(n), "amount": rng.uniform(1e3, 5e4, n)}
        )
        proba = rng.uniform(0.05, 0.95, n)
        res = weighted_pipeline(open_deals, proba=proba)
        assert list(res.columns) == ["expected", "n_deals"]
        assert len(res) == 1
        hand = float(np.sum(proba * open_deals["amount"].to_numpy()))
        assert res["expected"].iloc[0] == pytest.approx(hand)
        assert int(res["n_deals"].iloc[0]) == n

    def test_variance_and_interval_formula_exact(self):
        rng = np.random.default_rng(1)
        n = 60
        amount = rng.uniform(1e3, 5e4, n)
        open_deals = pd.DataFrame({"opp_id": np.arange(n), "amount": amount})
        proba = rng.uniform(0.05, 0.95, n)
        res = weighted_pipeline(open_deals, proba=proba, level=80)
        assert list(res.columns) == ["expected", "lo-80", "hi-80", "n_deals"]
        expected = float(np.sum(proba * amount))
        variance = float(np.sum(proba * (1 - proba) * amount**2))
        assert res["expected"].iloc[0] == pytest.approx(expected)
        hi = res["hi-80"].iloc[0]
        lo = res["lo-80"].iloc[0]
        assert hi == pytest.approx(expected + Z80 * np.sqrt(variance))
        assert lo == pytest.approx(expected - Z80 * np.sqrt(variance))
        # variance recovered from the symmetric interval matches the formula
        sd_recovered = (hi - expected) / Z80
        assert sd_recovered**2 == pytest.approx(variance)

    def test_interval_clamped_to_attainable_support(self):
        # one $100 deal at p=0.5: raw band is 50 ± Z80*50 = [-14, 114], but the
        # realized won-$ can only be 0 or 100, so lo floors to 0 and hi caps at
        # the $100 attainable maximum.
        open_deals = pd.DataFrame({"opp_id": [0], "amount": [100.0]})
        res = weighted_pipeline(open_deals, proba=[0.5], level=80)
        assert res["lo-80"].iloc[0] == 0.0
        assert res["hi-80"].iloc[0] == pytest.approx(100.0)

    def test_interval_uncapped_formula_when_support_is_wide(self):
        # many small deals: the band sits well inside [0, sum(amount)] so both
        # bounds equal the raw Normal-approx formula (no clamping).
        amounts = np.full(40, 10.0)
        p = np.full(40, 0.5)
        res = weighted_pipeline(
            pd.DataFrame({"opp_id": range(40), "amount": amounts}), proba=p, level=80
        )
        expected = float((p * amounts).sum())
        sd = float(np.sqrt((p * (1 - p) * amounts**2).sum()))
        assert res["lo-80"].iloc[0] == pytest.approx(expected - Z80 * sd)
        assert res["hi-80"].iloc[0] == pytest.approx(expected + Z80 * sd)

    def test_by_segment_grouping_sums_correctly(self):
        open_deals = pd.DataFrame(
            {
                "opp_id": [1, 2, 3, 4],
                "amount": [100.0, 200.0, 300.0, 400.0],
                "region": ["A", "A", "B", "B"],
            }
        )
        proba = np.array([0.5, 0.25, 0.8, 0.1])
        res = weighted_pipeline(open_deals, proba=proba, by="region")
        assert list(res.columns) == ["region", "expected", "n_deals"]
        by_region = res.set_index("region")["expected"]
        assert by_region["A"] == pytest.approx(0.5 * 100 + 0.25 * 200)
        assert by_region["B"] == pytest.approx(0.8 * 300 + 0.1 * 400)
        # segments partition the whole: their sum is the ungrouped total
        total = weighted_pipeline(open_deals, proba=proba)["expected"].iloc[0]
        assert by_region.sum() == pytest.approx(total)
        assert list(res["n_deals"]) == [2, 2]

    def test_multi_column_grouping(self):
        open_deals = pd.DataFrame(
            {
                "opp_id": [1, 2, 3, 4],
                "amount": [100.0, 200.0, 300.0, 400.0],
                "region": ["A", "A", "B", "B"],
                "product": ["x", "y", "x", "x"],
            }
        )
        proba = np.array([0.5, 0.5, 0.5, 0.5])
        res = weighted_pipeline(open_deals, proba=proba, by=["region", "product"])
        assert list(res.columns) == ["region", "product", "expected", "n_deals"]
        assert len(res) == 3  # (A,x),(A,y),(B,x)

    def test_scorer_supplies_probabilities(self):
        train = make_deals(n=3000, seed=1)
        open_deals = make_deals(n=400, seed=5, closed=False)
        scorer = DealScorer(seed=0).fit(train)
        res_scorer = weighted_pipeline(open_deals, scorer=scorer)
        p = scorer.predict_proba(open_deals).to_numpy()
        hand = float(np.sum(p * open_deals["amount"].to_numpy()))
        assert res_scorer["expected"].iloc[0] == pytest.approx(hand)

    def test_80pct_interval_brackets_monte_carlo(self):
        rng = np.random.default_rng(7)
        n = 150
        amount = rng.uniform(1e3, 5e4, n)
        proba = rng.uniform(0.1, 0.9, n)
        open_deals = pd.DataFrame({"opp_id": np.arange(n), "amount": amount})
        res = weighted_pipeline(open_deals, proba=proba, level=80)
        lo = res["lo-80"].iloc[0]
        hi = res["hi-80"].iloc[0]
        expected = res["expected"].iloc[0]

        sim_rng = np.random.default_rng(123)
        draws = sim_rng.random((40_000, n)) < proba
        totals = draws @ amount
        assert totals.mean() == pytest.approx(expected, rel=0.02)
        coverage = float(np.mean((totals >= lo) & (totals <= hi)))
        assert 0.77 < coverage < 0.83
        # Normal-approx bounds track the Monte-Carlo 10th/90th percentiles
        assert lo == pytest.approx(np.quantile(totals, 0.10), rel=0.01)
        assert hi == pytest.approx(np.quantile(totals, 0.90), rel=0.01)

    def test_amount_col_override(self):
        open_deals = pd.DataFrame({"opp_id": [1, 2], "acv": [100.0, 200.0]})
        res = weighted_pipeline(open_deals, proba=[0.5, 0.5], amount_col="acv")
        assert res["expected"].iloc[0] == pytest.approx(150.0)


class TestWeightedPipelineValidation:
    def test_requires_scorer_or_proba(self):
        open_deals = pd.DataFrame({"opp_id": [1], "amount": [100.0]})
        with pytest.raises(ForecastOSError, match="scorer"):
            weighted_pipeline(open_deals)

    def test_missing_amount_column_raises(self):
        open_deals = pd.DataFrame({"opp_id": [1], "amount": [100.0]})
        with pytest.raises(ForecastOSError, match="acv"):
            weighted_pipeline(open_deals, proba=[0.5], amount_col="acv")

    def test_proba_length_mismatch_raises(self):
        open_deals = pd.DataFrame({"opp_id": [1, 2], "amount": [100.0, 200.0]})
        with pytest.raises(ForecastOSError):
            weighted_pipeline(open_deals, proba=[0.5])

    def test_proba_out_of_range_raises(self):
        open_deals = pd.DataFrame({"opp_id": [1, 2], "amount": [100.0, 200.0]})
        with pytest.raises(ForecastOSError):
            weighted_pipeline(open_deals, proba=[0.5, 1.5])

    def test_missing_by_column_raises(self):
        open_deals = pd.DataFrame({"opp_id": [1], "amount": [100.0]})
        with pytest.raises(ForecastOSError, match="region"):
            weighted_pipeline(open_deals, proba=[0.5], by="region")


def test_weighted_pipeline_interval_stays_within_attainable_support():
    """hi-{level} never exceeds the group's total attainable amount (Σ amount)."""
    import pandas as pd

    from forecast_os.gtm.opportunities import weighted_pipeline

    # one big deal per segment: support is {0, amount}, so hi cannot exceed amount
    deals = pd.DataFrame(
        {"opp_id": [1, 2, 3], "amount": [200_000.0, 500_000.0, 30_000.0],
         "region": ["APAC", "AMER", "APAC"]}
    )
    proba = pd.Series([0.5, 0.5, 0.5], index=deals.index)
    out = weighted_pipeline(deals, proba=proba, by="region", level=80)
    totals = deals.groupby("region")["amount"].sum()
    for _, row in out.iterrows():
        assert row["lo-80"] >= 0.0
        assert row["hi-80"] <= totals[row["region"]] + 1e-6

    # ungrouped case too
    total = weighted_pipeline(deals, proba=proba, level=80)
    assert total["hi-80"].iloc[0] <= deals["amount"].sum() + 1e-6
    assert total["lo-80"].iloc[0] >= 0.0
