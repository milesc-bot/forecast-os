"""Tests for models/kalman.py: MLE Kalman filter forecaster with native intervals."""

import numpy as np
import pytest

from forecast_os.core.exceptions import ForecastOSError, NotFittedError
from forecast_os.core.registry import get_model, list_models
from forecast_os.core.types import to_panel
from forecast_os.datasets.synthetic import generate_series
from forecast_os.models.kalman import KalmanForecaster

H = 8


def _contract_panel():
    return generate_series(
        n_series=3, length=80, freq="D", trend=0.3, seasonality=7, season_amp=5.0,
        noise=0.8, seed=21,
    )


def _naive_mae(train, test):
    last = train.groupby("unique_id")["y"].last()
    return float((test["y"] - test["unique_id"].map(last)).abs().mean())


class TestKalmanForecaster:
    def test_registered_as_statistical(self):
        assert isinstance(get_model("kalman"), KalmanForecaster)
        listed = list_models(family="statistical")
        assert "kalman" in set(listed["name"])

    def test_invalid_model_raises(self):
        with pytest.raises(ForecastOSError):
            KalmanForecaster(model="structural")

    def test_params_stored_and_clone(self):
        model = KalmanForecaster(model="local_linear")
        assert model.get_params() == {"model": "local_linear"}
        clone = model.clone()
        assert type(clone) is KalmanForecaster
        assert clone.get_params() == model.get_params()

    def test_predict_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            KalmanForecaster().predict(3)

    def test_local_level_on_noisy_constant(self):
        rng = np.random.default_rng(1)
        y = 10.0 + 0.5 * rng.standard_normal(200)
        model = KalmanForecaster(model="local_level").fit(to_panel(y))
        yhat = model.predict(10)["yhat"].to_numpy()
        assert np.allclose(yhat, yhat[0]), "local level forecast must be flat"
        assert abs(yhat[0] - 10.0) < 0.5

    def test_local_linear_tracks_trend(self):
        rng = np.random.default_rng(2)
        n = 120
        y = 5.0 + 2.0 * np.arange(n, dtype=float) + 0.2 * rng.standard_normal(n)
        model = KalmanForecaster(model="local_linear").fit(to_panel(y))
        yhat = model.predict(10)["yhat"].to_numpy()
        truth = 5.0 + 2.0 * np.arange(n, n + 10, dtype=float)
        assert np.allclose(yhat, truth, rtol=0.05)

    def test_interval_width_grows_with_h(self):
        rng = np.random.default_rng(3)
        y = np.cumsum(rng.standard_normal(300))
        model = KalmanForecaster(model="local_level").fit(to_panel(y))
        pred = model.predict(20, level=[80])
        width = (pred["hi-80"] - pred["lo-80"]).to_numpy()
        assert np.all(np.diff(width) > 0), "interval width must grow with horizon"
        assert width[-1] > 1.2 * width[0]

    def test_local_linear_beats_naive_on_trend(self, trend_panel):
        train = trend_panel.groupby("unique_id").head(100)
        test = trend_panel.groupby("unique_id").tail(20)
        pred = KalmanForecaster(model="local_linear").fit(train).predict(20)
        merged = test.merge(pred, on=["unique_id", "ds"], validate="one_to_one")
        assert len(merged) == len(test)
        mae = float((merged["y"] - merged["yhat"]).abs().mean())
        assert mae < _naive_mae(train, test)

    def test_fitted_values_are_one_step_predictions(self):
        rng = np.random.default_rng(6)
        y = 10.0 + 0.5 * rng.standard_normal(100)
        model = KalmanForecaster(model="local_level").fit(to_panel(y))
        fitted = model.fitted_values()["fitted"].to_numpy()
        assert np.isnan(fitted[0])
        assert np.isfinite(fitted[1:]).all()
        # one-step predictions of a noisy constant should hug the level
        assert abs(np.nanmean(fitted) - 10.0) < 0.5

    @pytest.mark.parametrize("model", ["local_level", "local_linear"])
    def test_mle_is_scale_equivariant(self, model):
        """Fitted variances and forecasts must follow the units of ``y``.

        Through v0.8.0 the optimizer searched the *absolute* log-variances of R
        and Q inside a fixed ``(-30, 30)`` box, starting from
        ``log(var(diff(y)))``. Both are unit-carrying, so re-denominating a
        series (dollars instead of millions) walked the fit into the corner of
        the box: R and Q were clamped, the trend was killed off the point
        forecast and the intervals came out ~5x too narrow at ``c = 1e8`` and
        ~50x too wide at ``c = 1e-8`` -- silently, with no convergence check.

        The Gaussian state-space likelihood obeys
        ``nll(c*y; c^2 R, c^2 Q, c^2 P0) = nll(y; R, Q, P0) + (n - k)*log(c)``
        (the offset is one ``log(c)`` per *retained* prediction error, and the
        filter skips the ``k`` diffuse warm-up rows; either way it does not
        depend on the parameters), so the
        MLE is exactly equivariant: ``R/c^2``, ``Q/c^2``, ``yhat/c`` and
        ``sigma/c`` must not depend on ``c``. Fitting a unit-scale copy of the
        series and scaling the units-carrying quantities back restores that; the
        tolerances here only absorb float noise in ``y * c`` (the slope variance
        of ``local_linear`` sits at the degenerate zero boundary for this DGP,
        hence the absolute floor on the variance comparison).
        """
        rng = np.random.default_rng(11)
        n = 200
        base = (
            50.0
            + np.cumsum(rng.standard_normal(n))
            + 0.4 * np.arange(n)
            + 2.0 * rng.standard_normal(n)
        )
        var_floor = 1e-4 * float(np.var(np.diff(base)))
        ref_var = ref_fcst = None
        for c in (1e-8, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e8):
            fitted = KalmanForecaster(model=model).fit(to_panel(base * c))
            state = next(iter(fitted._series_state.values()))
            pred = fitted.predict(10, level=[80])
            var = np.concatenate([[state["R_"] / c**2], np.diag(state["Q_"]) / c**2])
            fcst = np.concatenate(
                [pred["yhat"].to_numpy() / c, (pred["hi-80"] - pred["lo-80"]).to_numpy() / c]
            )
            if ref_var is None:
                ref_var, ref_fcst = var, fcst
            else:
                np.testing.assert_allclose(var, ref_var, rtol=2e-2, atol=var_floor)
                np.testing.assert_allclose(fcst, ref_fcst, rtol=2e-3)

    def test_large_magnitude_series_is_not_pinned_to_the_variance_bounds(self):
        """A dollar-denominated series must not clamp R and Q on the search box.

        The v0.8.0 bound artifact was directly observable: for ``var(diff(y)) >
        exp(30)`` the fit stopped at the corner of the box with ``R == Q ==
        exp(30)`` -- 140 nats worse than the optimum -- rather than at the MLE.
        The box is now applied to the variance *ratios* to ``var(diff(y))``, so
        the fitted variances must track the series' own scale and leave real
        headroom inside the box.
        """
        rng = np.random.default_rng(12)
        y = 1e8 * (100.0 + np.cumsum(rng.standard_normal(150)))
        state = next(
            iter(KalmanForecaster(model="local_level").fit(to_panel(y))._series_state.values())
        )
        var_dy = float(np.var(np.diff(y)))
        assert var_dy > np.exp(30.0), "the DGP must exceed the old absolute upper bound"
        ratios = np.concatenate([[state["R_"]], np.diag(state["Q_"])]) / var_dy
        assert np.all(np.exp(-30.0) < ratios) and np.all(ratios < np.exp(30.0))
        assert np.all(ratios < 10.0), f"variances not tracking var(diff(y)): {ratios}"
        assert state["R_"] != state["Q_"][0, 0], "R and Q collapsed onto the same bound"

    def test_absurd_magnitude_does_not_raise_a_bare_overflowerror(self):
        """Rescaling must saturate, not raise an untyped Python exception.

        The unit-scale fit multiplies the fitted variances back by the square of
        the series scale. Written as ``scale ** 2`` that is a *float* power,
        which raises ``OverflowError: (34, 'Result too large')`` above ~1.3e154
        instead of returning inf -- so a 1e155-magnitude panel died with a bare
        builtin exception from inside the library (1e153 fitted fine: the
        threshold was invisible to the caller). Every other quantity in the
        filter saturates to inf, and a library must not leak an untyped
        OverflowError, so the multiplication saturates too: the point forecasts
        stay in the units of ``y`` and only the (genuinely unrepresentable)
        variances go to inf.
        """
        rng = np.random.default_rng(0)
        y = (10.0 + rng.standard_normal(40)) * 1e155
        fitted = KalmanForecaster().fit(to_panel(y))
        pred = fitted.predict(3)
        assert np.isfinite(pred["yhat"]).all()
        assert pred["yhat"].iloc[0] == pytest.approx(y[-1], rel=0.5)

    def test_default_on_contract_panel(self):
        df = _contract_panel()
        pred = KalmanForecaster().fit(df).predict(H, level=[80])
        assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
        assert (pred.groupby("unique_id").size() == H).all()
        for col in ("yhat", "lo-80", "hi-80"):
            assert np.isfinite(pred[col]).all()
        assert (pred["lo-80"] <= pred["yhat"]).all()
        assert (pred["yhat"] <= pred["hi-80"]).all()
