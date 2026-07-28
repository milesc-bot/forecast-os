"""Tests for finance/garch.py: GARCH11 MLE and the registered GARCHVolatility model."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from scipy.optimize import OptimizeResult, minimize

from forecast_os.core.exceptions import ForecastOSError, NotFittedError
from forecast_os.core.registry import _REGISTRY
from forecast_os.core.types import ID_COL, to_panel
from forecast_os.datasets.synthetic import generate_returns, generate_series
from forecast_os.finance import garch as garch_module
from forecast_os.finance.garch import GARCH11, GARCHVolatility

TRUE_OMEGA, TRUE_ALPHA, TRUE_BETA = 2e-6, 0.08, 0.85


def _regime_returns(seed: int = 12, n: int = 600) -> np.ndarray:
    """Returns with a hard volatility regime break — the case GARCH exists for.

    This shape is what used to defeat the optimizer: every L-BFGS-B start
    aborted in its line search and ``fit`` kept the best failed run.
    """
    rng = np.random.default_rng(seed)
    vol = np.where(np.arange(n) < n // 2, 0.005, 0.05)
    return rng.normal(0.0, 1.0, n) * vol


def _reference_nll(params, z2: np.ndarray, var0: float) -> float:
    """The GARCH(1,1) likelihood, re-derived here independently of garch.py.

    Non-stationary parameters are penalized rather than bounded so that this
    reference solves the same constrained problem the estimator does.
    """
    omega, alpha, beta = params
    sigma2 = np.empty(z2.size)
    sigma2[0] = var0
    for t in range(1, z2.size):
        sigma2[t] = omega + alpha * z2[t - 1] + beta * sigma2[t - 1]
    sigma2 = np.maximum(sigma2, 1e-12)
    value = 0.5 * float(np.sum(np.log(sigma2) + z2 / sigma2))
    persistence = alpha + beta
    if persistence > 0.999:
        value += 1e6 * (persistence - 0.999)
    return value


def _z_scale(r: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Mirror garch.py's internal demean + z-scale: returns (z2, scale, var0)."""
    centered = r - np.mean(r)
    scale = float(np.std(centered))
    z2 = (centered / scale) ** 2
    return z2, scale, float(np.mean(z2))


@pytest.fixture(scope="module")
def garch_r():
    df = generate_returns(
        n_series=1, length=1200, mu=0.0, garch=(TRUE_OMEGA, TRUE_ALPHA, TRUE_BETA), seed=13
    )
    return df["y"].to_numpy()


@pytest.fixture(scope="module")
def fitted(garch_r):
    return GARCH11().fit(garch_r)


class TestGARCH11:
    def test_fit_returns_self_and_attrs(self, garch_r, fitted):
        assert isinstance(fitted, GARCH11)
        assert fitted.omega_ > 0
        assert 0 <= fitted.alpha_ < 1
        assert 0 <= fitted.beta_ < 1
        assert np.isclose(fitted.mu_, np.mean(garch_r))
        assert isinstance(fitted.cond_vol_, np.ndarray)
        assert fitted.cond_vol_.shape == garch_r.shape
        assert (fitted.cond_vol_ > 0).all()

    def test_recovers_persistence(self, fitted):
        assert 0.8 <= fitted.alpha_ + fitted.beta_ <= 0.99

    def test_long_run_vol_within_50pct(self, fitted):
        true_lr = np.sqrt(TRUE_OMEGA / (1 - TRUE_ALPHA - TRUE_BETA))
        est_lr = np.sqrt(fitted.omega_ / (1 - fitted.alpha_ - fitted.beta_))
        assert 0.5 <= est_lr / true_lr <= 1.5

    def test_forecast_variance_monotone_toward_long_run(self, fitted):
        v = fitted.forecast_variance(30)
        assert v.shape == (30,)
        assert np.isfinite(v).all() and (v > 0).all()
        lr = fitted.omega_ / (1 - fitted.alpha_ - fitted.beta_)
        gaps = np.abs(v - lr)
        assert np.all(np.diff(gaps) <= 1e-12), "distance to long-run variance must shrink"
        diffs = np.diff(v)
        assert np.all(diffs >= -1e-15) or np.all(diffs <= 1e-15), "approach must be one-sided"

    def test_forecast_volatility_is_sqrt_of_variance(self, fitted):
        np.testing.assert_allclose(
            fitted.forecast_volatility(10), np.sqrt(fitted.forecast_variance(10))
        )

    def test_scale_equivariance(self, garch_r):
        g1 = GARCH11().fit(garch_r)
        g2 = GARCH11().fit(100.0 * garch_r)
        assert np.isclose(g2.alpha_, g1.alpha_, atol=1e-4)
        assert np.isclose(g2.beta_, g1.beta_, atol=1e-4)
        assert np.isclose(g2.omega_, 1e4 * g1.omega_, rtol=1e-3)
        np.testing.assert_allclose(g2.cond_vol_, 100.0 * g1.cond_vol_, rtol=1e-3)

    def test_forecast_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            GARCH11().forecast_variance(5)

    def test_fitted_params_beat_reference_parameter_vectors(self, garch_r, fitted):
        """Dead-optimizer guard: the fitted parameters must achieve an internal
        NLL no worse than a set of plausible hand-picked parameter vectors, i.e.
        the optimizer actually moved. Mirrors garch.py's z-scoring and var0 by
        formula and scores with an independently written likelihood.
        """
        z2, scale, var0 = _z_scale(garch_r)
        # fitted omega_ is reported on the original scale; map back to z-scale
        fitted_params = (fitted.omega_ / scale**2, fitted.alpha_, fitted.beta_)
        fitted_nll = _reference_nll(fitted_params, z2, var0)
        for x0 in ((0.05, 0.05, 0.90), (0.10, 0.10, 0.80), (0.30, 0.02, 0.95)):
            assert fitted_nll <= _reference_nll(x0, z2, var0) + 1e-9, (
                f"fitted NLL {fitted_nll:.2f} is worse than reference point {x0} "
                f"NLL {_reference_nll(x0, z2, var0):.2f}: optimizer did not improve"
            )

    def test_regime_switching_series_reaches_the_likelihood_optimum(self):
        """fit() never inspected ``res.success``, so on a volatility-regime
        series — where all three L-BFGS-B starts aborted with
        ABNORMAL_TERMINATION_IN_LNSRCH — it returned the best *failed* run.
        Those parameters were up to ~100 nats short of the MLE (alpha pinned
        near 0.5 instead of ~0.2). The estimator's own docstring promises
        conditional maximum likelihood, so the fitted point must match an
        independently multistarted optimum of the same constrained likelihood.
        """
        r = _regime_returns()
        fitted = GARCH11().fit(r)
        assert fitted.converged_ is True

        z2, scale, var0 = _z_scale(r)
        # omega ceiling mirrors garch.py's lam box: lam <= 10*var0/(1-0.999),
        # and omega = lam*(1-p) <= that same number at p -> 0.
        bounds = [(1e-8, 10.0 * var0 / (1.0 - 0.999)), (0.0, 0.998), (0.0, 0.998)]
        rng = np.random.default_rng(0)
        best = np.inf
        for _ in range(12):  # gradient-free, so independent of the fix's machinery
            alpha = rng.uniform(0.0, 0.5)
            beta = rng.uniform(0.0, 0.995 - alpha)
            omega = rng.uniform(1e-6, var0)
            res = minimize(
                _reference_nll,
                np.array([omega, alpha, beta]),
                args=(z2, var0),
                method="Nelder-Mead",
                bounds=bounds,
                options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 4000},
            )
            best = min(best, float(res.fun))

        fitted_params = (fitted.omega_ / scale**2, fitted.alpha_, fitted.beta_)
        fitted_nll = _reference_nll(fitted_params, z2, var0)
        assert fitted_nll <= best + 1e-4, (
            f"fitted NLL {fitted_nll:.4f} is {fitted_nll - best:.4f} nats worse "
            f"than the multistart optimum {best:.4f}: fit() is not returning the MLE"
        )

    def test_fit_is_stable_under_a_floating_point_perturbation(self):
        """Multiplying the returns by (1 + 1e-15) used to move alpha from 0.384
        to 0.133 on this series (and by 26% in volatility terms on the series in
        the audit), because which *failed* optimizer run happened to hold the
        lowest objective was decided by floating-point noise. Mathematically
        identical inputs must give mathematically identical parameters.
        """
        r = _regime_returns()
        base = GARCH11().fit(r)
        nudged = GARCH11().fit(r * (1.0 + 1e-15))
        assert np.isclose(nudged.alpha_, base.alpha_, rtol=1e-6, atol=1e-8)
        assert np.isclose(nudged.beta_, base.beta_, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(
            nudged.forecast_volatility(1), base.forecast_volatility(1), rtol=1e-6
        )

    def test_garch11_and_garchvolatility_agree_on_the_same_series(self):
        """Two public entry points onto the same estimator used to disagree on the
        same data — alpha 0.384 direct vs 0.136 through GARCHVolatility here,
        and volatility 0.0732 vs 0.0582 on the series in the audit — because
        GARCHVolatility's extra z-scoring nudged the optimizer into a different
        failed-optimization basin. They must now agree.
        """
        r = _regime_returns()
        direct = GARCH11().fit(r).forecast_volatility(3)
        wrapped = GARCHVolatility().fit(to_panel(r)).predict(3)["yhat"].to_numpy()
        np.testing.assert_allclose(wrapped, direct, rtol=1e-3)

    def test_non_convergence_raises_instead_of_returning_non_mle_params(self, monkeypatch):
        """The defect was silence: fit() compared only ``res.fun`` and never
        ``res.success``, so a run the optimizer had abandoned was reported as
        the MLE. When nothing converges — not even the gradient-free retry —
        fit() must fail loudly rather than return arbitrary parameters.
        """
        methods = []

        def stalled(fun, x0, **kwargs):
            methods.append(kwargs.get("method"))
            return OptimizeResult(
                x=np.asarray(x0, dtype=float),
                fun=123.0,
                success=False,
                message="ABNORMAL_TERMINATION_IN_LNSRCH",
            )

        monkeypatch.setattr(garch_module, "minimize", stalled)
        with pytest.raises(ForecastOSError, match="did not converge"):
            GARCH11().fit(np.random.default_rng(0).standard_normal(100))
        assert "Nelder-Mead" in methods, "a stalled gradient fit must be retried gradient-free"

    def test_near_integrated_fit_is_not_pinned_to_the_long_run_variance_bound(self):
        """The ``(1e-8, 10*var0)`` box was written for ``omega`` and then reused
        verbatim for ``lam``, the LONG-RUN variance — a far tighter restriction,
        since ``omega = lam*(1-p)``. On this near-integrated series the optimum
        parked exactly on ``lam == 10*var0`` and came back 0.0197 nats worse than
        an unconstrained search, reported as converged. ``lam`` must be free to
        reach whatever the old ``omega`` ceiling allowed, i.e. up to
        ``10*var0/(1-_MAX_PERSISTENCE)``.

        37.703335 is this series' wall-free optimum: it is what a wide-start
        multistart on this file's own ``_reference_nll`` reaches, and also what
        the pre-``(lam, p, q)`` estimator returned.
        """
        r = _regime_returns(n=200)
        fitted = GARCH11().fit(r)

        z2, scale, var0 = _z_scale(r)
        lam = (fitted.omega_ / scale**2) / (1.0 - fitted.alpha_ - fitted.beta_)
        assert lam > 10.0 * var0, f"long-run variance {lam:.6f} is stuck on the old omega wall"

        fitted_params = (fitted.omega_ / scale**2, fitted.alpha_, fitted.beta_)
        fitted_nll = _reference_nll(fitted_params, z2, var0)
        assert fitted_nll <= 37.703335 + 1e-4, (
            f"fitted NLL {fitted_nll:.6f} is worse than the wall-free optimum 37.703335"
        )

    def test_low_persistence_optimum_is_reachable_from_the_start_grid(self):
        """Every start had ``p`` in [0.90, 0.98], so a series whose optimum is
        weakly persistent never got a start in the right basin. On iid returns
        (no ARCH effect) fit() reported persistence 0.864 at NLL 47.273 when the
        optimum is 0.322 with beta = 0 at 46.719 — 0.55 nats short, flagged
        ``converged_ = True``. The start grid must span low persistence too.
        """
        r = np.random.default_rng(501).standard_normal(100)
        fitted = GARCH11().fit(r)

        z2, scale, var0 = _z_scale(r)
        fitted_params = (fitted.omega_ / scale**2, fitted.alpha_, fitted.beta_)
        fitted_nll = _reference_nll(fitted_params, z2, var0)
        optimum = _reference_nll((var0 * (1.0 - 0.322), 0.322, 0.0), z2, var0)
        assert fitted_nll <= optimum + 1e-4, (
            f"fitted NLL {fitted_nll:.5f} is {fitted_nll - optimum:.5f} nats worse "
            f"than the low-persistence optimum {optimum:.5f}"
        )
        assert fitted.alpha_ + fitted.beta_ < 0.6

    def test_converged_is_false_when_only_the_gradient_free_retry_worked(self, monkeypatch):
        """``converged_`` was the literal ``True``, so it carried no information
        and could not be used to screen a panel — even though fit() does return
        the weaker Nelder-Mead retry when every gradient start stalls. It must
        now report which of the two produced the parameters that were kept.
        """
        real_minimize = garch_module.minimize

        def stalled_gradient(fun, x0, **kwargs):
            if kwargs.get("method") == "L-BFGS-B":
                return OptimizeResult(
                    x=np.asarray(x0, dtype=float),
                    fun=float(fun(np.asarray(x0, dtype=float), *kwargs["args"])[0]),
                    success=False,
                    message="ABNORMAL_TERMINATION_IN_LNSRCH",
                )
            return real_minimize(fun, x0, **kwargs)

        monkeypatch.setattr(garch_module, "minimize", stalled_gradient)
        fitted = GARCH11().fit(_regime_returns(n=200))
        assert fitted.converged_ is False
        assert fitted.omega_ > 0

    def test_converged_is_true_on_an_ordinary_gradient_fit(self, fitted):
        """The other half of the contract: an ordinary series that L-BFGS-B
        handles must still report ``converged_ = True``.
        """
        assert fitted.converged_ is True

    def test_bad_inputs_raise(self):
        with pytest.raises(ValueError):
            GARCH11().fit(np.array([]))
        with pytest.raises(ValueError):
            GARCH11().fit(np.array([0.01, -0.02]))  # too short
        with pytest.raises(ValueError):
            GARCH11().fit(np.ones(50))  # constant
        g = GARCH11().fit(np.random.default_rng(0).standard_normal(100))
        with pytest.raises(ValueError):
            g.forecast_variance(0)


def _contract_panel():
    return generate_series(
        n_series=3, length=80, freq="D", trend=0.3, seasonality=7, season_amp=5.0,
        noise=0.8, seed=21,
    )


class TestGARCHVolatility:
    def test_registered_as_garch_financial(self):
        spec = _REGISTRY["garch"]
        assert spec.cls is GARCHVolatility
        assert spec.family == "financial"

    def test_contract_panel_predict(self):
        model = GARCHVolatility()
        df = _contract_panel()
        assert model.fit(df) is model
        pred = model.predict(8, level=[80])
        assert list(pred.columns) == [ID_COL, "ds", "yhat", "lo-80", "hi-80"]
        assert (pred.groupby(ID_COL).size() == 8).all()
        assert np.isfinite(pred[["yhat", "lo-80", "hi-80"]].to_numpy()).all()
        assert (pred["yhat"] > 0).all(), "volatility forecasts must be positive"
        assert (pred["lo-80"] <= pred["hi-80"]).all()

    def test_fitted_values_nan_at_t0(self):
        model = GARCHVolatility().fit(_contract_panel())
        fv = model.fitted_values()
        for _, g in fv.groupby(ID_COL):
            fitted = g["fitted"].to_numpy()
            assert np.isnan(fitted[0])
            assert np.isfinite(fitted[1:]).all()
            assert (fitted[1:] > 0).all()

    def test_vol_forecast_scales_with_series(self, garch_r):
        m1 = GARCHVolatility().fit(to_panel(garch_r))
        m2 = GARCHVolatility().fit(to_panel(50.0 * garch_r))
        y1 = m1.predict(5)["yhat"].to_numpy()
        y2 = m2.predict(5)["yhat"].to_numpy()
        np.testing.assert_allclose(y2, 50.0 * y1, rtol=1e-3)

    def test_predict_before_fit_raises(self):
        with pytest.raises(ForecastOSError):
            GARCHVolatility().predict(3)

    def test_clone_roundtrip(self):
        model = GARCHVolatility()
        clone = model.clone()
        assert type(clone) is GARCHVolatility
        assert clone.get_params() == model.get_params()


class TestGARCHVolatilityIntervals:
    """Regression: prediction intervals for a strictly positive quantity.

    ``GARCHVolatility`` did not override ``_predict_sigma``, so it inherited the
    base class's generic residual scale: ``BaseForecaster`` computes
    ``resid = y - state["fitted"]`` and takes an uncentered RMS. For every other
    model ``fitted`` is a fit of ``y``; here it is the conditional VOLATILITY,
    so the "residual" was approximately ``y`` itself and the interval half-width
    tracked the level of the series rather than any uncertainty about the
    volatility. On the contract panel that gave yhat=16.86 with lo-80=-125.37
    (interval sigma 110.98 vs mean y 117.36), and even on a natural zero-mean
    returns panel lo-80 came out negative — impossible for a volatility.
    """

    @staticmethod
    def _returns_panel():
        rng = np.random.default_rng(0)
        return to_panel(0.01 * rng.standard_normal(300), unique_id="asset-0")

    def test_lower_bound_positive_on_returns_panel(self):
        pred = GARCHVolatility().fit(self._returns_panel()).predict(3, level=[80])
        assert (pred["lo-80"] > 0).all()
        assert (pred["lo-80"] < pred["yhat"]).all()
        assert (pred["hi-80"] > pred["yhat"]).all()

    def test_lower_bound_positive_on_level_panel(self):
        pred = GARCHVolatility().fit(_contract_panel()).predict(5, level=[80])
        assert (pred["lo-80"] > 0).all()

    def test_interval_width_scales_with_volatility_not_with_the_level(self):
        """Shifting the series by a constant leaves the volatility interval alone."""
        df = self._returns_panel()
        shifted = df.copy()
        shifted["y"] = shifted["y"] + 100.0
        a = GARCHVolatility().fit(df).predict(3, level=[80])
        b = GARCHVolatility().fit(shifted).predict(3, level=[80])
        width_a = (a["hi-80"] - a["lo-80"]).to_numpy()
        width_b = (b["hi-80"] - b["lo-80"]).to_numpy()
        # a +100 level shift changes the volatility (and so its interval) by
        # far less than the ~200x the old y-residual scale would have given
        assert np.allclose(width_b, width_a, rtol=0.5)

    def test_interval_is_a_relative_band_around_the_volatility_forecast(self):
        """Half-width is z * yhat / sqrt(2n): the sd of a volatility estimate."""
        df = self._returns_panel()
        pred = GARCHVolatility().fit(df).predict(4, level=[80])
        z = float(stats.norm.ppf(0.9))
        expected = z * pred["yhat"].to_numpy() / np.sqrt(2 * len(df))
        got = (pred["hi-80"] - pred["yhat"]).to_numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-10)

    def test_constant_series_has_degenerate_interval(self):
        df = to_panel(np.full(60, 3.0), unique_id="flat")
        pred = GARCHVolatility().fit(df).predict(2, level=[80])
        assert (pred["yhat"] == 0.0).all()
        assert (pred["lo-80"] == 0.0).all() and (pred["hi-80"] == 0.0).all()

    @staticmethod
    def _garch_draw(rng, n, omega=2e-6, alpha=0.08, beta=0.90, burn=300):
        """A GARCH(1,1) sample plus the true conditional sigma of the next step."""
        total = n + burn + 1
        r = np.zeros(total)
        s2 = np.zeros(total)
        s2[0] = omega / (1.0 - alpha - beta)
        for t in range(1, total):
            s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
            r[t] = np.sqrt(s2[t]) * rng.standard_normal()
        return r[burn : burn + n], float(np.sqrt(s2[burn + n]))

    def test_interval_is_not_calibrated_and_must_not_claim_to_be(self):
        """The band under-covers badly at h=1 — the docstring must not promise more.

        The class docstring used to say the band "is exact only at h = 1, where
        sigma_{t+1} is known given the parameters" and that for h >= 2 it is
        "conservative (too narrow)" (self-contradictory: too narrow is
        ANTI-conservative). Both claims are false. ``yhat / sqrt(2n)`` is the
        asymptotic sd of a sample standard deviation of i.i.d. normals; it
        omits the estimation error in (omega, alpha, beta), which dominates
        here. Measured against the data-generating conditional volatility, a
        nominal 80% band covers ~50% at h = 1 — the horizon the docstring
        called exact. This test fails if the interval is ever widened into
        calibration (fine — then update the docstring) or if someone restores
        an exactness claim on top of these numbers.
        """
        rng = np.random.default_rng(11)
        hits = 0
        reps = 25
        for _ in range(reps):
            r, true_next_sigma = self._garch_draw(rng, 250)
            pred = GARCHVolatility().fit(to_panel(r)).predict(1, level=[80])
            assert pred["lo-80"][0] > 0  # the positivity fix still holds
            hits += bool(pred["lo-80"][0] <= true_next_sigma <= pred["hi-80"][0])
        coverage = hits / reps
        assert coverage < 0.70, f"nominal 80% band covered {coverage:.0%} at h=1"


def test_interval_metrics_against_y_are_inapplicable_to_this_model():
    """Pin the leaderboard caveat the class docstring now carries.

    ``compare(metrics=["coverage"])`` scores a band around a *volatility*
    against *returns*, so it asks a question the model does not answer: the
    band is a narrow strictly-positive interval around sigma while ``y``
    straddles zero, and coverage-80 lands at or near 0 (0.0 on this panel)
    where a point-forecast model like ``naive`` sits near nominal. This is a
    property of the metric, not a regression in the model — the docstring says
    so, and this test fails if the docstring's claim ever stops being true.
    """
    import pandas as pd

    from forecast_os.engine import ForecastEngine

    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame(
        {
            ID_COL: "a",
            "ds": pd.date_range("2020-01-01", periods=n, freq="D"),
            "y": 0.001 + 0.02 * rng.standard_normal(n),
        }
    )
    out = ForecastEngine(models=["garch", "naive"]).compare(
        df, h=6, metrics=["coverage"], level=[80]
    )
    cov_garch = float(out.loc["garch", "coverage-80"])
    cov_naive = float(out.loc["naive", "coverage-80"])
    assert cov_garch <= 0.2, f"garch coverage-80 was {cov_garch}"
    assert cov_garch < cov_naive
