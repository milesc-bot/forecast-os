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
