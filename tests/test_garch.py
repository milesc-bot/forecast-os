"""Tests for finance/garch.py: GARCH11 MLE and the registered GARCHVolatility model."""

from __future__ import annotations

import numpy as np
import pytest
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
