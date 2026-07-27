"""Tests for deal-level win-probability scoring and the probabilistic
pipeline forecast (gtm.opportunities)."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit

from forecast_os.core.exceptions import DataContractError, ForecastOSError, NotFittedError
from forecast_os.gtm.opportunities import DealScorer, _fit_platt, weighted_pipeline

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

    @pytest.mark.parametrize("dtype", ["boolean", "Int64", "Float64"])
    def test_nullable_target_with_pd_na_raises(self, dtype):
        """A pandas nullable target holding pd.NA must be rejected, not trained on.

        v0.8.0 dispatched on ``is_bool_dtype``, which is True for the nullable
        "boolean" extension dtype, and that branch converted straight to float
        with no finite check — turning pd.NA into a silent NaN. Every
        likelihood term then evaluated to NaN, L-BFGS-B never left x0, and fit
        SUCCEEDED with all-zero coefficients: a null model that scores every
        deal at p = 0.5, which weighted_pipeline happily reports as half the
        pipeline with an interval and no error anywhere. One deal not yet
        closed is a routine CRM shape, so this has to fail the same way the
        plain-float branch already did.
        """
        deals = make_deals(n=200, seed=1)
        deals["won"] = deals["won"].astype(dtype)
        deals.loc[0, "won"] = pd.NA
        with pytest.raises(DataContractError, match="won"):
            DealScorer(calibrate=False, seed=0).fit(deals)

    def test_nullable_target_without_na_still_accepted(self):
        """The NA guard must not reject a clean nullable column."""
        deals = make_deals(n=400, seed=1)
        plain = DealScorer(calibrate=False, seed=0).fit(deals)
        deals["won"] = deals["won"].astype("boolean")
        nullable = DealScorer(calibrate=False, seed=0).fit(deals)
        np.testing.assert_allclose(
            plain.coef_.to_numpy(), nullable.coef_.to_numpy()
        )


class TestCalibrationRegressions:
    """Regressions for the Platt calibrator (v0.8.0 audit MUST-4).

    v0.8.0 fit the calibrator against hard 0/1 targets on a single ~25%
    held-out split. On a separable split that objective has no finite maximum
    — the slope ran away to 1e3+ and pinned every win probability to 0 or 1 —
    so the DEFAULT ``calibrate=True`` made out-of-sample probabilities
    dramatically WORSE than ``calibrate=False``, and the pipeline intervals it
    fed were absurdly tight. The fix is Platt (1999)'s smoothed targets, which
    make the objective bounded under separation, plus cross-fitted scores and
    a minimum sample size so the map is estimated on real signal.
    """

    def test_separable_scores_do_not_blow_up_the_slope(self):
        """Perfectly separable held-out scores must still give a finite, mild map.

        With hard 0/1 targets this exact input returned a = 16.1 (and much
        larger on real fits) because the likelihood is unbounded when a
        threshold separates the classes. Smoothed targets are strictly inside
        (0, 1), so the loss diverges as a -> inf and the optimum is finite.
        """
        z = np.array([-2.0, -1.0, -0.5, 1.0, 2.0])
        y = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        a, b = _fit_platt(z, y)
        assert 0.0 < a < 5.0
        assert abs(b) < 5.0
        # and the resulting probabilities are not pinned to the boundary
        p = expit(a * z + b)
        assert p.min() > 0.01 and p.max() < 0.99

    @pytest.mark.parametrize("n,seed", [(100, 25), (100, 16), (100, 35)])
    def test_calibration_never_pins_probabilities_to_zero_or_one(self, n, seed):
        """A fitted scorer must not report near-certain probabilities for everyone.

        Under v0.8.0 these three fits produced _calib_a of 729, 302 and 77, so
        predict_proba collapsed to a 0/1 step function and out-of-sample log
        loss hit 5-6 nats (against 0.68 for simply predicting the base rate).
        Roughly one fit in twelve at n=100 landed here, so this was not an
        exotic corner.
        """
        scorer = DealScorer(calibrate=True, seed=0).fit(make_deals(n=n, seed=seed))
        assert scorer.calibrated_ is True  # the map really was fitted
        assert 0.0 < scorer._calib_a < 10.0
        p = scorer.predict_proba(make_deals(n=400, seed=77, closed=False)).to_numpy()
        # a genuine spread of probabilities, not a 0/1 step function
        assert float(np.mean((p > 0.05) & (p < 0.95))) > 0.5

    def test_calibrate_true_is_not_worse_than_calibrate_false_at_small_n(self):
        """The acceptance criterion: calibration must never degrade log loss.

        In v0.8.0, calibrate=True averaged ~1.8 nats of out-of-sample log loss
        at n=40 against ~0.46 for calibrate=False — four times worse than the
        setting it is supposed to improve on, and worse than predicting the
        base rate. When either outcome is too rare to estimate the map,
        calibration is skipped outright and the two settings must agree
        EXACTLY; otherwise the cross-fitted map must not cost more than a
        rounding error, and no single fit may blow up.

        (v0.9.0: the n < 100 half of the skip rule was removed — see
        ``test_calibration_runs_below_one_hundred_deals`` — so n=80 now
        calibrates and only the per-class floor keeps n=40 on the identity.)
        """
        test = make_deals(n=4000, seed=999)
        y_test = test["won"].to_numpy()
        base_ll = _log_loss(y_test, np.full(len(test), float(test["won"].mean())))

        for seed in (39, 11, 21):
            train = make_deals(n=40, seed=seed)  # < 20 of the rarer outcome
            with pytest.warns(UserWarning, match="calibrate=True"):
                cal = DealScorer(calibrate=True, seed=0).fit(train)
            unc = DealScorer(calibrate=False, seed=0).fit(train)
            assert cal.calibrated_ is False  # too few losses to calibrate
            np.testing.assert_allclose(
                cal.predict_proba(test).to_numpy(),
                unc.predict_proba(test).to_numpy(),
            )

        cal_lls, unc_lls = [], []
        for seed in range(1, 41):
            train = make_deals(n=100, seed=seed)
            cal = DealScorer(calibrate=True, seed=0).fit(train)
            unc = DealScorer(calibrate=False, seed=0).fit(train)
            cal_lls.append(_log_loss(y_test, cal.predict_proba(test).to_numpy()))
            unc_lls.append(_log_loss(y_test, unc.predict_proba(test).to_numpy()))
        mean_cal, mean_unc = float(np.mean(cal_lls)), float(np.mean(unc_lls))
        # calibration is at worst a rounding error, never a regression, and
        # both settings comfortably beat the constant base rate
        assert mean_cal < mean_unc * 1.01
        assert mean_cal < base_ll - 0.02
        # v0.8.0's worst fit over this same sweep was 6.40 nats, ~10x the
        # base rate; nothing may come remotely close to that now
        assert max(cal_lls) < 0.55

    def test_calibrated_flag_reports_whether_the_map_was_fitted(self):
        """calibrated_ must tell the truth about the small-sample fallback."""
        with pytest.warns(UserWarning, match="calibrate=True"):
            small = DealScorer(calibrate=True, seed=0).fit(make_deals(n=40, seed=1))
        assert small.calibrated_ is False
        big = DealScorer(calibrate=True, seed=0).fit(make_deals(n=1000, seed=1))
        assert big.calibrated_ is True
        assert big._calib_a != 1.0

    def test_calibration_runs_below_one_hundred_deals(self):
        """The n >= 100 calibration gate forfeited real accuracy; it is gone.

        The gate was introduced with the smoothed targets and cross-fitting
        that actually fixed the runaway slope, on the claim that below 100
        deals "the map is pure noise". Measurement does not support that. The
        map below 100 is a mild rescaling (fitted slope in [0.7, 1.5] at the
        default l2), and whenever the logistic is genuinely over-shrunk — the
        documented effect of raising l2, and precisely what post-hoc Platt
        scaling exists to repair — the gate silently forfeited 5-30% of
        out-of-sample log loss. It was also a cliff rather than a taper: fitting
        on 99 vs 100 of the same deals moved individual win probabilities by up
        to 0.26, and per-segment scorers either side of the line calibrated
        differently within one forecast.

        The per-class floor is kept: it is what stops a two-parameter map from
        being fitted on a handful of minority-class points, and it measured
        neutral. So at n=80 with plenty of both outcomes, calibration must run
        and must help when there is miscalibration to correct.
        """
        train = make_deals(n=80, seed=39)
        test = make_deals(n=4000, seed=999)
        y_test = test["won"].to_numpy()

        cal = DealScorer(calibrate=True, l2=10.0, seed=0).fit(train)
        unc = DealScorer(calibrate=False, l2=10.0, seed=0).fit(train)
        assert cal.calibrated_ is True
        assert 0.0 < cal._calib_a < 10.0  # a mild rescaling, not a runaway slope
        ll_cal = _log_loss(y_test, cal.predict_proba(test).to_numpy())
        ll_unc = _log_loss(y_test, unc.predict_proba(test).to_numpy())
        assert ll_cal < ll_unc

        # and no cliff at the old threshold: 99 vs 100 deals agree closely
        p99 = DealScorer(calibrate=True, l2=10.0, seed=0).fit(
            make_deals(n=100, seed=39).iloc[:99]
        ).predict_proba(test).to_numpy()
        p100 = DealScorer(calibrate=True, l2=10.0, seed=0).fit(
            make_deals(n=100, seed=39)
        ).predict_proba(test).to_numpy()
        assert float(np.max(np.abs(p99 - p100))) < 0.1

    def test_skipped_calibration_warns_and_a_real_calibration_does_not(self):
        """calibrate=True that cannot be honoured must say so, not go quiet.

        Previously a user who passed calibrate=True (the DEFAULT) below the
        threshold got no signal at all: zero warnings, and the only trace was
        calibrated_=False among ten public attributes. This codebase already
        warns for exactly this shape elsewhere — mstl.py drops unestimable
        seasonal periods with a warning, base.py drops unsupported exogenous
        columns with one — and the calibration skip must match.
        """
        train = make_deals(n=30, seed=1)  # only 11 losses: below the class floor
        with pytest.warns(UserWarning, match="ignoring calibrate=True") as rec:
            scorer = DealScorer(calibrate=True, seed=0).fit(train)
        assert scorer.calibrated_ is False
        msg = str(rec[0].message)
        assert "19 won" in msg and "11 lost" in msg  # names what was short
        assert "calibrated_ = False" in msg  # names the attribute to check

        # calibrate=False is an explicit choice, not a thwarted request
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            DealScorer(calibrate=False, seed=0).fit(train)
            big = DealScorer(calibrate=True, seed=0).fit(make_deals(n=1000, seed=1))
        assert big.calibrated_ is True

    def test_calibration_no_longer_costs_training_data(self):
        """With calibration on, the logistic is still fit on every deal.

        v0.8.0 threw away 25% of the training deals to hold out a calibration
        split, so calibrate=True and calibrate=False produced different
        coefficients from the same data for no benefit. Cross-fitting supplies
        honest out-of-fold scores without withholding anything, so the
        coefficients now match exactly and only the (a, b) map differs.
        """
        deals = make_deals(n=1000, seed=3)
        cal = DealScorer(calibrate=True, seed=0).fit(deals)
        unc = DealScorer(calibrate=False, seed=0).fit(deals)
        np.testing.assert_allclose(cal.coef_.to_numpy(), unc.coef_.to_numpy())
        assert cal.intercept_ == pytest.approx(unc.intercept_)

    def test_calibrated_pipeline_interval_is_not_absurdly_tight(self):
        """The runaway slope collapsed the pipeline band to nothing.

        With p pinned at 0/1 the Bernoulli variance p(1-p) is ~0 everywhere, so
        weighted_pipeline reported an 80% interval 0.016% wide that did not
        contain the realized booked amount. The band must stay wide enough to
        be honest about deal-level uncertainty.
        """
        train = make_deals(n=100, seed=25)  # _calib_a was 729 under v0.8.0
        open_deals = make_deals(n=300, seed=8, closed=False)
        scorer = DealScorer(calibrate=True, seed=0).fit(train)
        res = weighted_pipeline(open_deals, scorer=scorer, level=80)
        expected = float(res["expected"].iloc[0])
        width = float(res["hi-80"].iloc[0] - res["lo-80"].iloc[0])
        assert width / expected > 0.05


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


class TestSegmentsReconcileToTotal:
    """Regressions for null segment keys (v0.8.0 audit MUST-6).

    ``work.groupby(keys)`` inherited pandas' default ``dropna=True``, so every
    deal with a null ``by`` value vanished from the output: the segments no
    longer summed to the ungrouped total, silently and by an unbounded amount.
    An unassigned region/territory is an everyday CRM state, so this hit real
    forecasts.

    v0.9.0: the first fix raised on any null label, which restored the
    invariant by refusing to forecast an ordinary CRM export at all — and left
    the ``dropna=False`` added in the same change unreachable. The null-keyed
    deals are now KEPT as their own unlabelled segment (which is what makes the
    partition exact) and warned about. Segments must partition the whole; that
    is the invariant every test here pins.
    """

    OPEN = pd.DataFrame(
        {
            "opp_id": [1, 2, 3],
            "amount": [100.0, 200.0, 300.0],
            "region": ["A", None, "B"],
        }
    )
    PROBA = np.array([0.5, 1.0, 0.5])

    def _segments(self, deals, proba, by):
        with pytest.warns(UserWarning, match="null segment label"):
            return weighted_pipeline(deals, proba=proba, by=by)

    def test_null_segment_key_is_kept_as_its_own_segment(self):
        """The dropped $200 deal was the most certain in the pipeline (p=1.0)."""
        segments = self._segments(self.OPEN, self.PROBA, "region")
        total = weighted_pipeline(self.OPEN, proba=self.PROBA)
        assert segments["expected"].sum() == pytest.approx(
            float(total["expected"].iloc[0])
        )
        assert int(segments["n_deals"].sum()) == int(total["n_deals"].iloc[0]) == 3
        # the unlabelled deal is visible in the output with its own expected $
        unlabelled = segments[segments["region"].isna()]
        assert len(unlabelled) == 1
        assert float(unlabelled["expected"].iloc[0]) == pytest.approx(200.0)

    def test_null_segment_warning_names_the_column_and_the_amount_at_stake(self):
        with pytest.warns(UserWarning) as rec:
            weighted_pipeline(self.OPEN, proba=self.PROBA, by="region")
        msg = str(rec[0].message)
        assert "region" in msg
        assert "1 null" in msg
        assert "200" in msg  # the expected pipeline that used to vanish

    def test_nan_float_segment_key_also_reconciles(self):
        deals = pd.DataFrame(
            {"opp_id": [1, 2], "amount": [100.0, 200.0], "tier": [1.0, np.nan]}
        )
        segments = self._segments(deals, [0.5, 0.5], "tier")
        assert segments["expected"].sum() == pytest.approx(150.0)
        assert int(segments["n_deals"].sum()) == 2

    def test_null_in_any_of_several_by_columns_still_reconciles(self):
        deals = pd.DataFrame(
            {
                "opp_id": [1, 2],
                "amount": [100.0, 200.0],
                "region": ["A", "B"],
                "product": ["x", None],
            }
        )
        with pytest.warns(UserWarning, match="product"):
            segments = weighted_pipeline(deals, proba=[0.5, 0.5], by=["region", "product"])
        assert segments["expected"].sum() == pytest.approx(150.0)
        assert int(segments["n_deals"].sum()) == 2

    def test_categorical_by_column_with_a_null_label_works(self):
        """A CRM export read with dtype='category' is the common case, and the
        old error told the user to run a fillna that raises TypeError on it.

        There is no advice to follow now — the frame just forecasts — but the
        categorical path still has to reconcile, since ``to_numpy()`` on a
        Categorical is what feeds the groupby.
        """
        deals = pd.DataFrame(
            {
                "opp_id": [1, 2, 3],
                "amount": [100.0, 200.0, 300.0],
                "region": pd.Categorical(["A", "B", None]),
            }
        )
        segments = self._segments(deals, [0.5, 0.5, 0.5], "region")
        total = weighted_pipeline(deals, proba=[0.5, 0.5, 0.5])
        assert segments["expected"].sum() == pytest.approx(
            float(total["expected"].iloc[0])
        )
        assert int(segments["n_deals"].sum()) == 3
        assert segments["region"].isna().sum() == 1

    def test_fully_labelled_frame_warns_about_nothing(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            weighted_pipeline(
                self.OPEN.assign(region=["A", "C", "B"]), proba=self.PROBA, by="region"
            )

    def test_reconciliation_invariant_holds_once_nulls_are_labelled(self):
        """The invariant the bug broke: segments sum to the ungrouped total.

        The offending frame reconciles exactly as soon as the null label is
        filled, including the previously vanishing deal — so no deal and no
        dollar is lost anywhere in the grouping path.
        """
        filled = self.OPEN.copy()
        filled["region"] = filled["region"].fillna("unassigned")
        segments = weighted_pipeline(filled, proba=self.PROBA, by="region")
        total = weighted_pipeline(filled, proba=self.PROBA)
        assert segments["expected"].sum() == pytest.approx(
            float(total["expected"].iloc[0])
        )
        assert int(segments["n_deals"].sum()) == int(total["n_deals"].iloc[0]) == 3
        assert set(segments["region"]) == {"A", "B", "unassigned"}


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
