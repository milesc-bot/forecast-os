"""Tests for cohort retention: cohort_panel and the shifted-beta-geometric model."""

import numpy as np
import pandas as pd
import pytest

import forecast_os.gtm  # noqa: F401  (registers retention_sbg)
from forecast_os.core.exceptions import DataContractError, ForecastOSError
from forecast_os.core.registry import get_model, list_models
from forecast_os.core.types import ID_COL, TARGET_COL, TIME_COL, validate_panel
from forecast_os.datasets.synthetic import generate_series
from forecast_os.gtm import ShiftedBetaGeometric, cohort_panel


def _sbg_survival(alpha: float, beta: float, horizon: int) -> np.ndarray:
    """Exact sBG survival curve S(0..horizon) via the Fader-Hardie recursion."""
    p = np.zeros(horizon + 1)
    s = np.ones(horizon + 1)
    if horizon >= 1:
        p[1] = alpha / (alpha + beta)
        s[1] = 1.0 - p[1]
    for t in range(2, horizon + 1):
        p[t] = (beta + t - 2) / (alpha + beta + t - 1) * p[t - 1]
        s[t] = s[t - 1] - p[t]
    return s


def _retention_panel(alpha=1.2, beta=3.5, horizon=16, cohorts=("c1", "c2", "c3")):
    """Panel of identical exact sBG retention curves, ds = integer age 0..horizon."""
    s = _sbg_survival(alpha, beta, horizon)
    frames = [
        pd.DataFrame({ID_COL: c, TIME_COL: np.arange(horizon + 1), TARGET_COL: s})
        for c in cohorts
    ]
    return pd.concat(frames, ignore_index=True)


class TestCohortPanel:
    def _table(self):
        return pd.DataFrame(
            {
                "cohort": ["2024-01"] * 3 + ["2024-02"] * 2,
                "age": [0, 1, 2, 0, 1],
                "retained": [100, 80, 64, 200, 150],
                "n": [100, 100, 100, 200, 200],
            }
        )

    def test_counts_divided_by_cohort_size(self):
        panel = cohort_panel(self._table(), "cohort", "age", "retained", n_customers_col="n")
        validate_panel(panel)
        assert list(panel.columns) == [ID_COL, TIME_COL, TARGET_COL]
        c1 = panel[panel[ID_COL] == "2024-01"]
        assert list(c1[TIME_COL]) == [0, 1, 2]
        assert list(c1[TARGET_COL]) == [1.0, 0.8, 0.64]
        c2 = panel[panel[ID_COL] == "2024-02"]
        assert list(c2[TARGET_COL]) == [1.0, 0.75]

    def test_rates_passed_through_when_no_size_col(self):
        tbl = pd.DataFrame(
            {"cohort": ["a", "a"], "age": [0, 1], "rate": [1.0, 0.7]}
        )
        panel = cohort_panel(tbl, "cohort", "age", "rate")
        assert list(panel[TARGET_COL]) == [1.0, 0.7]

    def test_non_monotone_curves_allowed(self):
        tbl = pd.DataFrame(
            {"cohort": ["a"] * 3, "age": [0, 1, 2], "rate": [1.0, 0.8, 0.85]}
        )
        panel = cohort_panel(tbl, "cohort", "age", "rate")
        assert list(panel[TARGET_COL]) == [1.0, 0.8, 0.85]

    def test_out_of_range_rate_raises(self):
        tbl = pd.DataFrame({"cohort": ["a", "a"], "age": [0, 1], "rate": [1.0, 1.2]})
        with pytest.raises(DataContractError, match=r"\[0, 1\]"):
            cohort_panel(tbl, "cohort", "age", "rate")

    def test_missing_column_raises(self):
        with pytest.raises(DataContractError, match="missing"):
            cohort_panel(self._table(), "cohort", "age", "nope")


class TestShiftedBetaGeometric:
    def test_registered_in_gtm_family(self):
        frame = list_models(family="gtm")
        assert "retention_sbg" in set(frame["name"])
        assert isinstance(get_model("retention_sbg"), ShiftedBetaGeometric)

    def test_recovers_known_parameters(self):
        alpha, beta = 1.2, 3.5
        model = ShiftedBetaGeometric().fit(_retention_panel(alpha=alpha, beta=beta))
        params = model.cohort_params()
        assert list(params.columns) == [ID_COL, "alpha", "beta", "pooled"]
        assert not params["pooled"].any()
        np.testing.assert_allclose(params["alpha"], alpha, rtol=0.1)
        np.testing.assert_allclose(params["beta"], beta, rtol=0.1)

    def test_forecast_monotone_nonincreasing_in_unit_interval(self):
        panel = _retention_panel()
        model = ShiftedBetaGeometric().fit(panel)
        pred = model.predict(6)
        last = panel.groupby(ID_COL)[TARGET_COL].last()
        for uid, g in pred.groupby(ID_COL):
            y = g.sort_values(TIME_COL)["yhat"].to_numpy()
            assert np.all(np.diff(y) <= 1e-12), f"{uid} forecast not nonincreasing"
            assert np.all((y >= 0.0) & (y <= 1.0))
            assert y[0] <= last[uid] + 1e-9  # continues from last observed retention

    def test_short_cohort_uses_pooled_parameters(self):
        long_panel = _retention_panel(cohorts=("c1", "c2"))
        short = pd.DataFrame(
            {ID_COL: "tiny", TIME_COL: [0, 1], TARGET_COL: [1.0, 0.8]}
        )
        model = ShiftedBetaGeometric().fit(pd.concat([long_panel, short], ignore_index=True))
        params = model.cohort_params().set_index(ID_COL)
        assert bool(params.loc["tiny", "pooled"])
        assert params.loc["tiny", "alpha"] == model.pooled_params_[0]
        assert params.loc["tiny", "beta"] == model.pooled_params_[1]
        assert not bool(params.loc["c1", "pooled"])
        # the 2-point cohort still gets a valid monotone forecast
        pred = model.predict(4)
        y = pred[pred[ID_COL] == "tiny"].sort_values(TIME_COL)["yhat"].to_numpy()
        assert np.all(np.diff(y) <= 1e-12)
        assert np.all((y >= 0.0) & (y <= 0.8 + 1e-9))

    def test_out_of_range_values_raise_at_fit(self):
        generic = generate_series(n_series=2, length=30, freq="D", seed=3)
        with pytest.raises(ForecastOSError, match="retention fractions"):
            ShiftedBetaGeometric().fit(generic)

    def test_gapped_cohort_ages_raise_at_fit(self):
        """Regression: gapped ages silently mapped observations to wrong ages."""
        panel = _retention_panel(cohorts=("c1", "c2"))
        gapped = pd.DataFrame(
            {ID_COL: "gappy", TIME_COL: [0, 1, 3, 4], TARGET_COL: [1.0, 0.8, 0.6, 0.55]}
        )
        with pytest.raises(DataContractError, match=r"consecutive.*'gappy'"):
            ShiftedBetaGeometric().fit(pd.concat([panel, gapped], ignore_index=True))

    def test_all_short_cohorts_raise(self):
        short = pd.DataFrame(
            {
                ID_COL: ["a", "a", "b", "b"],
                TIME_COL: [0, 1, 0, 1],
                TARGET_COL: [1.0, 0.8, 1.0, 0.7],
            }
        )
        with pytest.raises(ForecastOSError, match="at least"):
            ShiftedBetaGeometric().fit(short)


class TestCohortAgeAnchoredOnDs:
    """Regression: the cohort age came from ``y[0]``, not from ``ds``.

    ``_strip_leading_one`` only treated the first value as the age-0 anchor
    when it was >= 1.0. A real cohort with activation churn (5% of the
    signups never make it to their first renewal, so age-0 retention is
    0.95) therefore had its ENTIRE curve read one age too early: S(1) was
    fitted against the age-0 observation, and ``predict`` projected from
    T = last_age + 1. On an exact sBG(1, 4) cohort that recovered
    alpha=3.22, beta=20.39 instead of (1, 4) and the 5-step forecast drifted
    monotonically to -14.5% and was still diverging.

    ``ds`` is the cohort age (that is what ``cohort_panel`` builds and what
    the consecutive-age guard already promised), so ``ds`` — not the value
    at ``y[0]`` — is what anchors the curve. Age 0 is the acquisition
    anchor and its value is the scale the survival curve is measured
    against; a curve that starts at age 1 simply has no age-0 observation,
    which implies full activation (S(0) = 1.0).
    """

    ALPHA, BETA, LAST_AGE = 1.0, 4.0, 10

    def _cohort(self, activation=0.95, first_age=0):
        """One cohort: ``activation * S_sBG(t)`` at ages ``first_age..LAST_AGE``."""
        ages = np.arange(first_age, self.LAST_AGE + 1)
        s = _sbg_survival(self.ALPHA, self.BETA, self.LAST_AGE)
        return pd.DataFrame({ID_COL: "c1", TIME_COL: ages, TARGET_COL: activation * s[ages]})

    def _truth(self, h, activation=0.95):
        s = _sbg_survival(self.ALPHA, self.BETA, self.LAST_AGE + h)
        return activation * s[self.LAST_AGE + 1 :]

    def test_sub_one_age0_retention_recovers_the_true_curve(self):
        model = ShiftedBetaGeometric().fit(self._cohort(activation=0.95))
        params = model.cohort_params().iloc[0]
        assert not params["pooled"]
        np.testing.assert_allclose(params["alpha"], self.ALPHA, rtol=0.05)
        np.testing.assert_allclose(params["beta"], self.BETA, rtol=0.05)
        yhat = model.predict(5)["yhat"].to_numpy()
        np.testing.assert_allclose(yhat, self._truth(5), rtol=0.01)

    def test_activation_rate_is_a_scale_not_an_age_shift(self):
        """Scaling the whole curve must move the forecast, never the parameters."""
        full = ShiftedBetaGeometric().fit(self._cohort(activation=1.0))
        partial = ShiftedBetaGeometric().fit(self._cohort(activation=0.95))
        for name in ("alpha", "beta"):
            np.testing.assert_allclose(
                partial.cohort_params()[name], full.cohort_params()[name], rtol=1e-6
            )
        np.testing.assert_allclose(
            partial.predict(4)["yhat"].to_numpy(),
            0.95 * full.predict(4)["yhat"].to_numpy(),
            rtol=1e-9,
        )

    def test_curve_starting_at_age_one_assumes_full_activation(self):
        """No age-0 row means S(0)=1.0 is implied; the panel is not altered."""
        panel = self._cohort(activation=1.0, first_age=1)
        model = ShiftedBetaGeometric().fit(panel)
        params = model.cohort_params().iloc[0]
        np.testing.assert_allclose(params["alpha"], self.ALPHA, rtol=0.05)
        np.testing.assert_allclose(params["beta"], self.BETA, rtol=0.05)
        np.testing.assert_allclose(
            model.predict(5)["yhat"].to_numpy(), self._truth(5, activation=1.0), rtol=0.01
        )
        # the implied anchor is internal: fitted_values still mirrors the input
        fv = model.fitted_values()
        assert list(fv[TIME_COL]) == list(panel[TIME_COL])
        np.testing.assert_allclose(fv[TARGET_COL], panel[TARGET_COL])

    def test_curve_starting_at_a_later_age_raises(self):
        """ds 3..10 was silently fitted as if the first point were S(1)."""
        with pytest.raises(DataContractError, match=r"age 0.*age 1.*'c1'.*age 3"):
            ShiftedBetaGeometric().fit(self._cohort(first_age=3))

    def test_later_first_age_message_names_the_left_truncation_remedy(self):
        """The advice used to be unfollowable for the commonest cause.

        A cohort observed from age 3 because the earlier periods are missing
        from the warehouse ALREADY has ages relative to acquisition, so
        "re-index the ages relative to each cohort's acquisition period" asks
        for something the user has already done. The remedy that works is
        subtracting the first observed age: sBG is closed under left
        truncation (S(a+k)/S(a) is itself sBG with beta + a), so the forecast
        is exactly right and only the reported beta shifts. The message must
        say so.
        """
        with pytest.raises(DataContractError) as exc:
            ShiftedBetaGeometric().fit(self._cohort(first_age=3))
        msg = str(exc.value)
        assert "truncat" in msg
        assert "subtract 3" in msg
        assert "beta + 3" in msg

    def test_dead_cohort_uses_the_pooled_fallback_instead_of_aborting(self):
        """Regression: one all-zero cohort raised and killed the whole panel.

        A cohort with 0 retention at age 0 is not malformed data — it is a
        channel where nobody activated — and the correct forecast for it is
        0.0. The previous guard rejected it on an arbitrary ``<= 1e-9`` cut
        (1e-10 aborted the panel, 1e-8 was accepted and returned exactly the
        zeros the rejected cohort would have produced), taking five healthy
        cohorts down with it. A cohort whose scale is unusable now degrades
        to the pooled fallback, exactly like a cohort whose MLE fails.
        """
        healthy = _retention_panel(cohorts=("c1", "c2", "c3"))
        dead = pd.DataFrame(
            {ID_COL: "dead", TIME_COL: [0, 1, 2, 3], TARGET_COL: [0.0, 0.0, 0.0, 0.0]}
        )
        model = ShiftedBetaGeometric().fit(pd.concat([healthy, dead], ignore_index=True))
        params = model.cohort_params().set_index(ID_COL)
        assert bool(params.loc["dead", "pooled"])
        assert params.loc["dead", "alpha"] == model.pooled_params_[0]
        yhat = model.predict(4)
        np.testing.assert_allclose(yhat[yhat[ID_COL] == "dead"]["yhat"], 0.0)
        # the healthy cohorts are untouched by their dead neighbour
        alone = ShiftedBetaGeometric().fit(healthy)
        np.testing.assert_allclose(
            alone.pooled_params_, model.pooled_params_, rtol=1e-9
        )
        for uid in ("c1", "c2", "c3"):
            np.testing.assert_allclose(
                params.loc[uid, "alpha"],
                alone.cohort_params().set_index(ID_COL).loc[uid, "alpha"],
                rtol=1e-9,
            )

    def test_pooled_curves_are_scale_free(self):
        """Pooling averages normalized curves, so activation cannot skew them."""
        panel = pd.concat(
            [
                self._cohort(activation=1.0).assign(unique_id="full"),
                self._cohort(activation=0.6).assign(unique_id="weak"),
            ],
            ignore_index=True,
        )
        model = ShiftedBetaGeometric().fit(panel)
        np.testing.assert_allclose(model.pooled_params_[0], self.ALPHA, rtol=0.05)
        np.testing.assert_allclose(model.pooled_params_[1], self.BETA, rtol=0.05)


class TestPooledPriorIsBounded:
    """Regression: pooling normalized curves was unbounded above.

    Dividing each cohort by its age-0 anchor makes the pooled curve
    scale-free, which is right, but ``y[1:] / y[0]`` exceeds 1 whenever a
    cohort's retention rises out of age 0 — trial-to-paid conversion, a late
    first payment, an age-0 snapshot taken before the period closed. Those
    values are not survival probabilities, and the position-wise MEAN let a
    single such cohort dominate: 11 exact sBG(1, 4) cohorts plus one trial
    cohort produced a pooled "survival" curve starting at 1.108 and pooled
    params (4.26, 39.7) instead of (1, 4) — a prior every short cohort in the
    panel borrows. ``_sbg_mle`` cannot catch it because it clips negative
    churn increments to zero weight and reports success.

    A curve whose normalized values exceed 1 says its age-0 value is not the
    cohort's scale, so it cannot inform a survival prior; it is left out of
    the pooled curve, which is then bounded in [0, 1] by construction. On
    clean data nothing is excluded and the pooled fit is unchanged (binomial
    retention can never exceed its own anchor: measured 0 exclusions over 480
    simulated cohorts at n=20..200 customers).
    """

    ALPHA, BETA, LAST_AGE = 1.0, 4.0, 10

    def _clean(self, uid):
        s = _sbg_survival(self.ALPHA, self.BETA, self.LAST_AGE)
        return pd.DataFrame(
            {ID_COL: uid, TIME_COL: np.arange(self.LAST_AGE + 1), TARGET_COL: s}
        )

    def _trial(self, uid):
        """20% activate at age 0, the rest convert by age 1, then churn."""
        vals = [0.20, 0.90, 0.75, 0.66, 0.60, 0.56, 0.53, 0.50, 0.48, 0.46, 0.44]
        return pd.DataFrame(
            {ID_COL: uid, TIME_COL: np.arange(self.LAST_AGE + 1), TARGET_COL: vals}
        )

    @pytest.mark.parametrize("n_clean", [11, 5, 1])
    def test_a_cohort_rising_out_of_age_zero_cannot_poison_the_prior(self, n_clean):
        frames = [self._clean(f"c{i}") for i in range(n_clean)] + [self._trial("trial")]
        model = ShiftedBetaGeometric().fit(pd.concat(frames, ignore_index=True))
        np.testing.assert_allclose(model.pooled_params_[0], self.ALPHA, rtol=0.05)
        np.testing.assert_allclose(model.pooled_params_[1], self.BETA, rtol=0.05)

    def test_pooled_curve_stays_a_survival_curve_when_most_cohorts_rise(self):
        """Even a majority of anchor-violating cohorts must not lift it above 1."""
        frames = [self._clean(f"c{i}") for i in range(2)]
        frames += [self._trial(f"t{i}") for i in range(7)]
        model = ShiftedBetaGeometric().fit(pd.concat(frames, ignore_index=True))
        np.testing.assert_allclose(model.pooled_params_[0], self.ALPHA, rtol=0.05)
        np.testing.assert_allclose(model.pooled_params_[1], self.BETA, rtol=0.05)

    def test_short_cohorts_borrow_the_clean_prior(self):
        """The point of the prior: young cohorts must not inherit the garbage."""
        frames = [self._clean(f"c{i}") for i in range(6)] + [self._trial("trial")]
        young = pd.DataFrame({ID_COL: "young", TIME_COL: [0, 1], TARGET_COL: [1.0, 0.79]})
        frames.append(young)
        model = ShiftedBetaGeometric().fit(pd.concat(frames, ignore_index=True))
        params = model.cohort_params().set_index(ID_COL)
        assert bool(params.loc["young", "pooled"])
        truth = _sbg_survival(self.ALPHA, self.BETA, 6)
        yhat = model.predict(4)
        got = yhat[yhat[ID_COL] == "young"].sort_values(TIME_COL)["yhat"].to_numpy()
        expected = 0.79 * truth[2:6] / truth[1]
        assert np.abs(got - expected).max() < 0.01


class TestThresholdsCountCallerRows:
    """Regression: the synthetic age-0 anchor was counted by the thresholds.

    ``fit`` materializes an age-0 row of 1.0 for cohorts written from age 1 so
    that every cohort takes one code path. That row is bookkeeping, not an
    observation, but ``pooled_threshold`` and ``min_train_size`` counted it —
    so the same four-period curve was shrunk toward the pooled prior when
    written from age 0 and fitted on its own noisy 4-point MLE when written
    from age 1, which is exactly the failure ``pooled_threshold`` exists to
    prevent (the verifier measured 38% worse forecast MAE over 60 simulated
    panels). Thresholds documented in observations must count the caller's
    rows.
    """

    def _four_row_cohorts(self, first_age):
        s = _sbg_survival(1.0, 4.0, 6)
        ages = np.arange(first_age, first_age + 4)
        vals = s[ages] if first_age == 0 else s[1:5]
        frames = [
            pd.DataFrame({ID_COL: f"k{i}", TIME_COL: ages, TARGET_COL: vals})
            for i in range(6)
        ]
        return pd.concat(frames, ignore_index=True)

    @pytest.mark.parametrize("first_age", [0, 1])
    def test_four_observation_cohorts_are_shrunk_either_way(self, first_age):
        model = ShiftedBetaGeometric(pooled_threshold=5).fit(
            self._four_row_cohorts(first_age)
        )
        assert model.cohort_params()["pooled"].all()

    def test_row_counts_stay_attached_to_their_own_cohort(self):
        """A mixed, shuffled panel must not mis-pair counts with cohorts."""
        spec = {"zeta": (1, 3), "alpha": (0, 4), "mu": (1, 8), "beta": (0, 12)}
        s = _sbg_survival(1.0, 4.0, 12)
        frames = [
            pd.DataFrame(
                {
                    ID_COL: uid,
                    TIME_COL: np.arange(first, first + n),
                    TARGET_COL: s[np.arange(first, first + n)],
                }
            )
            for uid, (first, n) in spec.items()
        ]
        panel = pd.concat(frames[::-1], ignore_index=True).sample(
            frac=1.0, random_state=1
        )
        model = ShiftedBetaGeometric(pooled_threshold=5).fit(panel)
        params = model.cohort_params().set_index(ID_COL)
        for uid, (_, n) in spec.items():
            assert bool(params.loc[uid, "pooled"]) is (n < 5), uid

    def test_row_counts_survive_a_categorical_unique_id(self):
        """A Categorical id groups in category order, not lexicographic order.

        Materializing an implied anchor concatenates an object-dtype id
        column, which turns a Categorical ``unique_id`` into object — so the
        cohorts re-sort between the caller's frame and the fitted frame. The
        row counts are paired by cohort, not by position, so a CRM export
        with ``dtype='category'`` cannot silently swap which cohort is
        shrunk.
        """
        spec = {"zeta": (1, 3), "alpha": (0, 12)}
        s = _sbg_survival(1.0, 4.0, 12)
        panel = pd.concat(
            [
                pd.DataFrame(
                    {
                        ID_COL: uid,
                        TIME_COL: np.arange(first, first + n),
                        TARGET_COL: s[np.arange(first, first + n)],
                    }
                )
                for uid, (first, n) in spec.items()
            ],
            ignore_index=True,
        )
        panel[ID_COL] = pd.Categorical(
            panel[ID_COL], categories=["zeta", "alpha"], ordered=True
        )
        params = (
            ShiftedBetaGeometric(pooled_threshold=5)
            .fit(panel)
            .cohort_params()
            .set_index(ID_COL)
        )
        assert bool(params.loc["zeta", "pooled"]) is True
        assert bool(params.loc["alpha", "pooled"]) is False

    @pytest.mark.parametrize("first_age", [0, 1])
    def test_min_train_size_counts_caller_rows_either_way(self, first_age):
        """Two-row cohorts are too short for a pooled fit at either start age."""
        vals = [1.0, 0.8] if first_age == 0 else [0.8, 0.64]
        frames = [
            pd.DataFrame(
                {ID_COL: f"t{i}", TIME_COL: [first_age, first_age + 1], TARGET_COL: vals}
            )
            for i in range(3)
        ]
        with pytest.raises(ForecastOSError, match="at least"):
            ShiftedBetaGeometric().fit(pd.concat(frames, ignore_index=True))


class TestRetentionContractEquivalent:
    """Mirror of tests/test_contract.py run on a retention-shaped panel.

    ``retention_sbg`` rejects non-retention data by design, so the generic
    contract panel cannot exercise it; this suite applies the identical
    assertions on a valid retention panel instead.
    """

    H = 5

    def _panel(self):
        frames = []
        for uid, (a, b) in {"c1": (0.8, 2.5), "c2": (1.2, 3.5), "c3": (2.0, 6.0)}.items():
            s = _sbg_survival(a, b, 16)
            frames.append(
                pd.DataFrame({ID_COL: uid, TIME_COL: np.arange(17), TARGET_COL: s})
            )
        return pd.concat(frames, ignore_index=True)

    def test_contract_on_retention_panel(self):
        model = get_model("retention_sbg")
        df = self._panel()
        fitted = model.fit(df)
        assert fitted is model, "fit() must return self"

        pred = model.predict(self.H, level=[80])
        assert list(pred.columns[:3]) == [ID_COL, TIME_COL, "yhat"]
        assert {"lo-80", "hi-80"} <= set(pred.columns)

        counts = pred.groupby(ID_COL).size()
        assert set(counts.index) == set(df[ID_COL].unique())
        assert (counts == self.H).all()
        assert np.isfinite(pred["yhat"]).all()
        assert (pred["lo-80"] <= pred["hi-80"] + 1e-9).all()

        last_train = df.groupby(ID_COL)[TIME_COL].max()
        for uid, g in pred.groupby(ID_COL):
            assert (g[TIME_COL] > last_train[uid]).all()
            assert g[TIME_COL].is_monotonic_increasing

        clone = model.clone()
        assert type(clone) is type(model)
        assert clone.get_params() == model.get_params()
        pred2 = clone.fit(df).predict(self.H)
        pd.testing.assert_series_equal(
            pred["yhat"], pred2["yhat"], check_exact=False, atol=1e-6
        )

    def test_predict_before_fit_raises(self):
        with pytest.raises(ForecastOSError):
            get_model("retention_sbg").predict(3)

    @pytest.mark.parametrize("bad_h", [0, -1])
    def test_predict_rejects_nonpositive_h(self, bad_h):
        model = get_model("retention_sbg").fit(self._panel())
        with pytest.raises(ValueError):
            model.predict(bad_h)

    @pytest.mark.parametrize("bad_level", [[0], [150]])
    def test_predict_rejects_bad_level(self, bad_level):
        model = get_model("retention_sbg").fit(self._panel())
        with pytest.raises(ValueError):
            model.predict(3, level=bad_level)

    def test_fitted_values_shape(self):
        model = get_model("retention_sbg").fit(self._panel())
        fv = model.fitted_values()
        assert list(fv.columns) == [ID_COL, TIME_COL, TARGET_COL, "fitted"]
        assert len(fv) == len(self._panel())
        # in-sample fits are near-exact on noiseless model-generated curves
        resid = (fv[TARGET_COL] - fv["fitted"]).dropna()
        assert np.abs(resid).max() < 0.05
