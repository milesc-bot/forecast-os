"""Tests for finance/garch.py: GARCH11 MLE and the registered GARCHVolatility model."""

from __future__ import annotations

import numpy as np
import pytest

from forecast_os.core.exceptions import ForecastOSError, NotFittedError
from forecast_os.core.registry import _REGISTRY
from forecast_os.core.types import ID_COL, to_panel
from forecast_os.datasets.synthetic import generate_returns, generate_series
from forecast_os.finance.garch import GARCH11, GARCHVolatility

TRUE_OMEGA, TRUE_ALPHA, TRUE_BETA = 2e-6, 0.08, 0.85


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

    def test_fitted_params_beat_every_optimizer_start(self, garch_r, fitted):
        """Dead-optimizer guard: the fitted parameters must achieve an internal
        NLL no worse than every hard-coded optimizer start, i.e. the optimizer
        actually moved. Mirrors garch.py's z-scoring, var0, starts, and _nll
        (including the persistence penalty) by formula.
        """
        r = garch_r - np.mean(garch_r)
        scale = float(np.std(r))
        z2 = (r / scale) ** 2
        var0 = float(np.mean(z2))

        def nll(omega: float, alpha: float, beta: float) -> float:
            sigma2 = np.empty(z2.size)
            sigma2[0] = var0
            for t in range(1, z2.size):
                sigma2[t] = omega + alpha * z2[t - 1] + beta * sigma2[t - 1]
            sigma2 = np.maximum(sigma2, 1e-12)
            val = 0.5 * float(np.sum(np.log(sigma2) + z2 / sigma2))
            persistence = alpha + beta
            if persistence > 0.999:
                val += 1e6 * (persistence - 0.999)
            return val

        # fitted omega_ is reported on the original scale; map back to z-scale
        fitted_nll = nll(fitted.omega_ / scale**2, fitted.alpha_, fitted.beta_)
        starts = ((0.05, 0.05, 0.90), (0.10, 0.10, 0.80), (0.30, 0.02, 0.95))
        for x0 in starts:
            assert fitted_nll <= nll(*x0) + 1e-9, (
                f"fitted NLL {fitted_nll:.2f} is worse than start {x0} "
                f"NLL {nll(*x0):.2f}: optimizer did not improve on its start"
            )

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
